import os
import glob
import argparse
import json
import random

import cv2
import openslide
import numpy as np
import pandas as pd
from datetime import datetime

from minotaur.wsi.parser import WSIParser
from minotaur.wsi.utilities import TissueDetect, visualise_wsi_tiling, StainNormalizer


def parse_wsi(args, wsi_path):
    
    try:
        wsi = openslide.OpenSlide(wsi_path)
    except openslide.lowlevel.OpenSlideError as e:
        return False
    
    try:
        args.base_mag = wsi.properties[
            openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    except:
        print('gregggggggggggggggg')
        pass
    
    try:
        args.mpp = wsi.properties[
            openslide.PROPERTY_NAME_MPP_X]
    except:
        pass

    #print(f'Base mag: {args.base_mag}, mpp: {args.mpp}')
    #ds = int(int(args.base_mag) / 10) 
    ds_factors = [int(d) for d in wsi.level_downsamples]
    level = ds_factors.index(32) if 32 in ds_factors else ds_factors.index(int(ds_factors[-1]))
    #level = ds_factors.index(ds)
    #print(f'Downsample: {ds} \nLevel: {level}')
    detector = TissueDetect(wsi)
    thumb = detector.tissue_thumbnail
    tis_mask = detector.detect_tissue()
    border = detector.border()

    cv2.imwrite(os.path.join(
        args.vis_path, args.name+'_thumb.png'), thumb)
    cv2.imwrite(os.path.join(
        args.vis_path, args.name+'_mask.png'), tis_mask)

    if args.normalize:
        normalizer = StainNormalizer(args.sn_target, args.sn_method)
    else:
        normalizer = None

    #parser = WSIParser(wsi, args.tile_dims, border, args.mag_level)
    parser = WSIParser(wsi, args.tile_dims, border, args.mag_level, normalizer)

    num = parser.tiler(args.stride)
    print('Tiles: {}'.format(num))

    parser.filter_tissue(
        tis_mask,
        label=1,
        threshold=args.filter_threshold)
    print(f'Filtered tiles: {parser.number}')

    visualise_wsi_tiling(
            wsi,
            parser,
            os.path.join(args.vis_path,
                         args.name+'_tiling.png'),
            viewing_res=level
            )
     
    if args.sample:
        parser.sample_tiles(args.sample)
        #print('tilessssss',len(parser._tiles))
        print(f'Sampled tiles: {parser.number}')

    if args.parser != 'tiler':
        func = parser.extract_features(
            args.parser,
            args.model_path,
            downsample=args.downsample,
            normalize=args.normalize
        )
        #print("Feature extracted!")
    else:
        func = parser.extract_tiles(args.normalize)

    if args.parser != 'tiler' and args.database == 'lmdb':
        parser.to_lmdb(
            func,
            os.path.join(args.tile_path, args.name), 
            map_size = args.map_size)
        print("parsed to lmdb")
    elif args.parser != 'tiler' and args.database == 'rocksdb':
            parser.to_rocksdb(func, os.path.join(args.tile_path, args.name))
            print("parsed to rocksdb")
    elif args.parser != 'tiler' and args.database == 'disk':
        parser.feat_to_disk(func, os.path.join(args.tile_path, args.name))
        print("parsed to feat_to_disk")
    else:
        print('savingggg')
        parser.save(func, os.path.join(args.tile_path, args.name),label_dir=True)
    
    return True
    
    
if __name__=='__main__':

    ap=argparse.ArgumentParser()

    ap.add_argument('-wp','--wsi_path',
            required=True, help='whole slide image directory')

    ap.add_argument('-sp','--save_path',
            required=True, help='directoy to write tiles and features')

    ap.add_argument('-p', '--parser',
            required=True, help='wsi parsing approach')

    ap.add_argument('-mp', '--model_path', default=None,
            help='path to trained model if parser is not tiler')

    ap.add_argument('-td','--tile_dims', default=512, type=int,
            help='dimensions of tiles')

    ap.add_argument('-s','--stride', default=512, type=int,
            help='distance to step across WSI')

    ap.add_argument('-ml','--mag_level', default=0, type=int,
            help='magnification level of tiling')

    ap.add_argument('-ds','--downsample', required=False, type=int,
            help='downsample tiles to resolution inbetween levels')

    ap.add_argument('-n','--normalize', default=False, type=bool,
            help='downsample tiles to resolution inbetween levels')

    ap.add_argument('-sm', '--sn_method', default='macenko',
            help='Stain normalization method (options: macenko, vahadane, reinhard)')

    ap.add_argument('-st', '--sn_target', default=None,
            help='Path to the target image for stain normalization')

    ap.add_argument('-sa','--sample', required=False, type=int,
            help='sample number')

    ap.add_argument('-db','--database', default=None,
            help='store tiles/features in database. Options are lmdb, rocksdb, or disk')

    ap.add_argument('-mz', '--map_size', default=int(4e9), type=int,
            help='map_size of lmdb database if in use')

    ap.add_argument('-ft','--filter_threshold', default=0.5, type=float,
            help='threshold to filter tiles based on background')

    ap.add_argument('-dp','--dataset_csv', default=None,
            help='Path to CSV of slides and corresponding labels')

    args=ap.parse_args()
    import torch    
    print('avaliable', torch.cuda.is_available())

    dir_ = 'tiles' if args.parser == 'tiler' else 'features'
    args.tile_path = os.path.join(args.save_path, dir_)
    args.vis_path = os.path.join(args.save_path, 'vis')
    os.makedirs(args.tile_path, exist_ok=True)
    os.makedirs(args.vis_path, exist_ok=True)
    
    # Print arguments
    print("\nParsed arguments:\n")

    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    print("\n")
   
    # Save parsed arguments as json for future reference
    json_file_path = os.path.join(args.save_path, 'arguments.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    
    # If dataset csv is supplied
    if args.dataset_csv is not None:
        saved = pd.read_csv(args.dataset_csv)
        if 'ID' in saved.columns:
            if saved['ID'].notna().any():
                saved = list(saved['ID'])
            else:
                print(f"Column 'ID' exists in {args.dataset_csv} but is empty.")
        else:
            print(f"Column 'ID' does not exist in {args.dataset_csv}.")

    errors = []
    success = dict(name = [], time = [])

    wsi_paths=glob.glob(os.path.join(args.wsi_path,'*'))
    #print(len(wsi_paths))
    #random.seed(4)
    #random.shuffle(wsi_paths)
    #random.shuffle(wsi_paths)
    #n = len(wsi_paths)
    #k = int(0.1*n)
    #index=random.sample(range(1,n),k)
    #wsi_paths = list(np.array(wsi_paths)[index])
    check = [
        'TCGA-85-8582-01Z-00-DX1.ac10318b-f56e-4897-8355-076366fe8581',
        'TCGA-CV-7236-01Z-00-DX1.0755A8D7-0E79-471B-8C3B-C4CBD2733FD3',
        'TCGA-CV-A460-01Z-00-DX1.EE85F1E3-B39D-43B6-A329-ED132E7B137A',
        'TCGA-CJ-4643-01Z-00-DX1.50F38125-1825-4B66-A171-4C92279E306D',
        'TCGA-AA-3667-01Z-00-DX1.28dc3612-1c43-4727-a134-698cc4315dc3',
        'TCGA-E9-A22B-01Z-00-DX1.01692f98-bd7c-4b02-990c-53560276baa0',
        'TCGA-F5-6814-01Z-00-DX1.25b7eec3-80bc-43c2-9969-78f343c55bed',
        'TCGA-CZ-5989-01Z-00-DX1.c6fd2257-4742-4c4e-ade2-995f54f1cfa8',
        'TCGA-AR-A0TY-01Z-00-DX1.089EB6B4-9921-441D-84F3-C64C97AFC3B4',
        'TCGA-CV-A45W-01Z-00-DX1.6123D9EE-BB0B-479E-9CCF-345AAD567E9C',
        'TCGA-F7-A61S-01Z-00-DX1.BD41A8AB-B10D-4020-A00D-47A3A3827955',
        'TCGA-F7-A620-01Z-00-DX1.E0DA4A79-6F9B-4F7C-912E-4A4514DF8F49',
        'TCGA-T7-A92I-01Z-00-DX2.87A6A24E-42E3-4107-AE91-369C0D168556',
        'TCGA-BP-5170-01Z-00-DX1.ae43bef7-3d81-4f69-be37-b4958bf79939',
        'TCGA-BP-4985-01Z-00-DX1.46854f1c-2f91-4990-b0ca-c76ff09bd835',
        'TCGA-CN-5367-01Z-00-DX1.5b09e54e-4140-4709-bc60-e201f9a72b24',
        'TCGA-YL-A9WI-01Z-00-DX1.7BEF88F1-9093-41C0-834C-7DD4A332455E',
        'TCGA-AN-A0FK-01Z-00-DX1.8966A1D5-CE3A-4B08-A1F6-E613BEB1ABD1',
        'TCGA-85-8070-01Z-00-DX1.54e7edf1-28d3-4fd5-bab3-951e58620386',
        'TCGA-CQ-7072-01Z-00-DX1.5932E8B0-B5B3-4ECC-AC61-A85FEE443536',
        'TCGA-49-4494-01Z-00-DX1.2bf10e85-465a-4b17-abd7-3565f0e538a5',
        'TCGA-EI-6510-01Z-00-DX1.e1209172-b10e-4dca-a714-16960a498cf4',
        'TCGA-37-3789-01Z-00-DX1.d7a4923e-988e-45e9-a64e-30c983740614',
        'TCGA-98-8020-01Z-00-DX1.fcf8e956-c413-40ae-9f09-610403e70fb3',
        'TCGA-QK-AA3K-01Z-00-DX1.A7F73A27-D890-4E05-86A8-26B6F3AA2332',
        'TCGA-HC-7738-01Z-00-DX1.CB18FC7E-9960-4639-9D95-B1EA15D1E8E1',
        'TCGA-EJ-7314-01Z-00-DX1.d0eb7da6-3259-475f-b81c-146dca9b5613',
        'TCGA-D8-A27R-01Z-00-DX1.F6E2FD1C-0666-4788-8D95-A76D15907270',
        'TCGA-F7-A50J-01Z-00-DX1.D754F90E-8C16-4EFB-97A4-27591CBC250D',
        'TCGA-A6-5660-01Z-00-DX1.b254e383-a889-4b73-8f91-8580c8285754',
        'TCGA-CV-7238-01Z-00-DX1.125B2906-3C04-46F7-887B-044BDA3724A2',
        'TCGA-CV-7252-01Z-00-DX1.01B710B3-B69F-4DE2-BF2B-06048EDA7A14',
        'TCGA-BA-A6DD-01Z-00-DX1.5B67DD6F-811C-4DE2-894E-C8FE4CF7FBE7',
        'TCGA-49-4506-01Z-00-DX3.a05a3969-bbef-48d8-86de-f51c8870afd6',
        'TCGA-18-3414-01Z-00-DX1.B76AF657-7A93-4F33-BC1A-2E8BF652FC17',
        'TCGA-YB-A89D-01Z-00-DX1.8FCB30EE-4982-42D8-98CA-0A21945CADDD',
        'TCGA-55-A490-01Z-00-DX1.07D77502-7216-4C23-9BB8-8DE99AECC920',
        'TCGA-AA-3521-01Z-00-DX1.9d6be975-7be1-4f6c-99db-5101369c6624',
        'TCGA-BH-A1FU-01Z-00-DX1.3D195750-8C9E-48C6-92EC-44AC6AB56267',
        'TCGA-DQ-5629-01Z-00-DX1.A5FB326B-4EC1-4446-B98A-392126168863',
        'TCGA-X4-A8KS-01Z-00-DX2.E7A0D551-4DF3-47B0-885A-181E798FC36B',
        'TCGA-49-4512-01Z-00-DX7.1f758560-85e2-4cf9-a81a-317536d96bcc',
        'TCGA-CV-7407-01Z-00-DX1.F26EB9EC-6AB2-403D-88CC-3330F37542E8',
        'TCGA-OK-A5Q2-01Z-00-DX2.C828A160-87DF-4625-A8C5-2057F61D54F4',
        'TCGA-CV-7090-01Z-00-DX1.6DF21B0C-86F4-4ECB-A599-4F001A288E3B',
        'TCGA-CK-5914-01Z-00-DX1.dfa6d814-6ddb-4058-a236-d57303cbfbe9',
        'TCGA-BP-5010-01Z-00-DX1.e906e855-5dc0-4c28-85e3-a0ce1af27f5d',
        'TCGA-A3-3349-01Z-00-DX1.206c2817-1b93-4fdf-8128-3f6e3b4935b6',
        'TCGA-KK-A6E8-01Z-00-DX1.5A8673FF-44A5-4383-86E8-A830205C99C7',
        'TCGA-BP-4162-01Z-00-DX1.cb017dff-4a35-4e33-92a7-2006973ad6bf',
        'TCGA-D6-8568-01Z-00-DX1.70A20D5E-6628-4F2E-B808-D20BEB7DB3BF',
        'TCGA-BH-A18P-01Z-00-DX1.C66642D3-BE44-4D65-B4AE-C1C3D959D22C',
        'TCGA-CJ-4908-01Z-00-DX1.3B9E62AC-B67F-4AAC-9255-8A1EB3A78473',
        'TCGA-AA-3866-01Z-00-DX1.f93457c3-abaa-4268-84e2-394d7c1aa523',
        'TCGA-D5-6538-01Z-00-DX1.fab8da8e-1e0a-4fb2-987c-9792c05d5a3a',
        'TCGA-93-7347-01Z-00-DX1.a22b7163-07bd-4f6a-97c7-f1561c4d73cb',
        'TCGA-BP-5006-01Z-00-DX1.98cb6ac9-bb30-4b71-aa66-0d08f80ecd55',
        'TCGA-IB-AAUN-01Z-00-DX1.70F5AFF8-FECD-4BEE-AD00-E03F05D29B58',
        'TCGA-KK-A8IK-01Z-00-DX1.FB5F2457-F503-4A0F-8911-C87F2251333C',
        'TCGA-G9-7521-01Z-00-DX1.6b2a4c47-db98-44fe-b5f8-c96ec398f37a',
        'TCGA-GM-A2D9-01Z-00-DX1.AF4BF2DD-05FB-400B-A1BC-6E7C9B9DDF05',
        'TCGA-CK-4950-01Z-00-DX1.03dcc4c2-2b63-45a2-8561-bf18193202b5',
        'TCGA-68-8250-01Z-00-DX1.a8669a85-9262-4c37-b61c-ca7019a3677c',
        'TCGA-BP-4337-01Z-00-DX1.5c88395f-317b-4cc5-8945-16c7cdb0876e',
        'TCGA-A2-A04U-01Z-00-DX1.06D17357-46A8-4DC3-A22B-2F4EB6EE3F79',
        'TCGA-BP-4346-01Z-00-DX1.e2f7b4ce-fbe8-482e-822e-ef428731f68b',
        'TCGA-B0-4712-01Z-00-DX1.b584d650-4bdd-452c-9926-18a1a8388d91',
        'TCGA-B0-4688-01Z-00-DX1.cb882331-49f1-4662-b142-0e9836993eb4',
        'TCGA-A2-A0YL-01Z-00-DX1.69A438C7-B1E0-4990-B1E6-586C551DC79C',
        'TCGA-BA-A8YP-01Z-00-DX1.03B95E63-2639-49F6-B8DD-AC2C2FA872B1',
        'TCGA-P3-A6T7-01Z-00-DX1.B150E785-D8CA-4D1A-95B7-D678E5A66AFC',
        'TCGA-IB-AAUU-01Z-00-DX1.83137319-8229-4682-8BD3-9B9A5C6C997A',
        'TCGA-AR-A0TT-01Z-00-DX1.127C2A4F-62AC-4E83-99DB-13519BD2949D',
        'TCGA-BP-4329-01Z-00-DX1.da54fc65-7cb6-4265-8d45-b373a59fa1da',
        'TCGA-CN-5366-01Z-00-DX1.dcc14c44-6dd1-48a1-a21c-736c6b9551e1',
        'TCGA-44-4112-01Z-00-DX1.5380b3b4-0e24-4c6c-aa8c-298c5fbe068f',
        'TCGA-CV-A6JD-01Z-00-DX1.AEEF2C26-FBCB-43A9-AC34-E9C1570A1BBC',
        'TCGA-B4-5844-01Z-00-DX1.64CF8817-E960-45EC-8073-8228DE0486C0',
        'TCGA-85-8288-01Z-00-DX1.0e68535f-a1f3-450c-b8fa-a859f2269da8',
        'TCGA-49-4501-01Z-00-DX1.9c4bf212-5e32-449e-8710-cf7a3594156e',
        'TCGA-NJ-A55A-01Z-00-DX1.42C356FE-52A3-4A60-886B-75F84ADBC534',
        'TCGA-CM-6676-01Z-00-DX1.dcc2bf23-ecaa-4952-8485-fc609af66298',
        'TCGA-EJ-5509-01Z-00-DX1.d5ddb1aa-3fd9-468b-abfa-97b908695a30',
        'TCGA-DV-5566-01Z-00-DX1.715a6d3c-1172-4f06-8077-32d2a4efb5b3',
        'TCGA-EW-A6SD-01Z-00-DX1.32D8240E-2076-492B-BB95-300A9FCA96E7',
        'TCGA-KU-A66S-01Z-00-DX1.195F55BC-4F8F-4039-8E12-4B18C53A8B04',
        'TCGA-IQ-A61K-01Z-00-DX1.92AE10A1-0F59-48DD-A259-6C1B9BD63689',
        'TCGA-BP-4998-01Z-00-DX1.f3ea98d5-3807-47d3-bcd6-5501d7679b88',
        'TCGA-CV-6933-01Z-00-DX1.2AF4A57B-EEED-4781-8C7B-74C542D60605',
        'TCGA-LL-A9Q3-01Z-00-DX1.94FA76F5-008A-401B-BFDC-817804BAE5F6',
        'TCGA-85-7697-01Z-00-DX1.b33aa550-81b0-4fab-8243-d041b0beaab5',
        'TCGA-G4-6295-01Z-00-DX1.9e7ae22f-daac-42cb-a879-bcf505d1c725',
        'TCGA-B0-4824-01Z-00-DX1.c454abac-df1a-4610-be22-49358c436f8d',
        'TCGA-CM-6168-01Z-00-DX1.96af6eb2-9d51-4671-baf8-1a73d0c66869',
        'TCGA-AZ-6608-01Z-00-DX1.40d9f93f-f7d8-4138-9af1-bb579c53194b',
        'TCGA-18-3408-01Z-00-DX1.B4AFC08A-7460-4EE6-B033-629C6A6CA6E8',
        'TCGA-BP-4765-01Z-00-DX1.eeb51a80-0a8d-428e-ab84-92769b558a76',
        'TCGA-B6-A0WZ-01Z-00-DX1.6CFB236E-36F5-43D6-8DE3-C4ECBD3C14C6',
        'TCGA-KK-A6DY-01Z-00-DX1.A58E02B6-F093-43EE-B06C-975C4171E8A1',
        'TCGA-BP-4967-01Z-00-DX1.25615528-2d42-4b48-b086-2e851f4fee6e',
        'TCGA-VP-A87J-01Z-00-DX1.31A1E162-71C5-4F15-B9C5-AFAB41506F86',
        'TCGA-AG-3586-01Z-00-DX1.38d7b4ce-9f18-48e6-9988-a3ef43b3d646',
        'TCGA-21-5783-01Z-00-DX1.3FAB28DF-5748-42D5-8257-3D440C4FB5FB',
        'TCGA-18-4086-01Z-00-DX1.1D06B771-F978-4763-8D4E-7B540E02C55A',
        'TCGA-46-3766-01Z-00-DX1.5870b2ed-78f0-4f68-b01e-2ef6d5353b57',
        'TCGA-77-8144-01Z-00-DX1.5194d537-c942-45f3-967b-eb213646ff24',
        'TCGA-AZ-4315-01Z-00-DX1.1a2c2771-3e59-47c3-b380-42110c545e6b',
        'TCGA-E2-A14T-01Z-00-DX1.61B4C988-6D75-447B-A5F4-9DE92CEACC9F',
        'TCGA-AA-3856-01Z-00-DX1.973974e7-fcfe-4866-bc0c-50645c6c304b',
        'TCGA-AA-3672-01Z-00-DX1.6cc142eb-e77f-4c09-a6ac-e85470221812',
        'TCGA-49-4512-01Z-00-DX5.7198ce36-1fae-4da1-9f26-b7f43cf01133',
        'TCGA-AA-3955-01Z-00-DX1.e0eea910-db79-4797-b9db-bb8bfe35306d',
        'TCGA-E9-A1RI-01Z-00-DX1.3259644B-0888-4B18-B16F-397D55167EAD',
        'TCGA-B2-3924-01Z-00-DX1.751b1107-fa18-4df5-89ae-42d06caf9b74',
        'TCGA-AY-A71X-01Z-00-DX1.68F9BC0F-1D60-4AEF-9083-509387038F03',
        'TCGA-CQ-7065-01Z-00-DX1.083ed2a3-9074-4e48-b950-7255ca54ce77',
        'TCGA-KK-A8I8-01Z-00-DX1.9C138C77-7DD7-4EE0-A34C-D4B629E18087',
        'TCGA-EJ-5502-01Z-00-DX1.9caa4792-06a2-431d-bec3-8b17f41453fe',
        'TCGA-FB-AAPY-01Z-00-DX1.06F4ED05-E333-48E1-91B9-345ACA41A4DF',
        'TCGA-86-7701-01Z-00-DX1.a8a6e71e-9fa9-42c6-a186-0ac7526e9960',
        'TCGA-AR-A24S-01Z-00-DX1.317C3B19-0D0F-4315-AC2D-C741F09DD775',
        'TCGA-BP-5004-01Z-00-DX1.46d61a35-7e4f-4fce-9785-8efe255852f0',
        'TCGA-CQ-6222-01Z-00-DX1.78668666-ed6f-49d8-a243-5635cb61875d',
        'TCGA-HC-A9TH-01Z-00-DX1.13526861-B926-4CF5-9E99-D4A8FFFBBE7C',
        'TCGA-AA-3681-01Z-00-DX1.576342cf-0f40-404a-b3c5-b33103f86777',
        'TCGA-49-AARN-01Z-00-DX1.E13AA041-978E-427D-AFA5-D3180224A03A',
        'TCGA-BH-A1ET-01Z-00-DX1.05C126CD-CC10-44BF-9A68-6CDDE97272B2',
        'TCGA-D8-A1XS-01Z-00-DX1.E5C78E5A-947C-499D-9E95-619FEEC63E69',
        'TCGA-EJ-5526-01Z-00-DX1.70486c91-c1cd-4478-8c90-b103dacda0ae',
        'TCGA-CM-5868-01Z-00-DX1.70f2e193-248d-4bf9-a875-49c314223f70',
        'TCGA-BP-4988-01Z-00-DX1.b037f17b-8fc7-46b2-adae-8e1140606fb7',
        'TCGA-BP-4801-01Z-00-DX1.69e0e4d9-dc88-4010-acec-15183034b356',
        'TCGA-BH-A0DG-01Z-00-DX1.D3C4F57F-608F-46AA-A978-B558117C9DFB',
        'TCGA-CV-6938-01Z-00-DX1.8D0880C1-2E01-4954-95C0-6F15FF0F0B0A',
        'TCGA-BP-4352-01Z-00-DX1.fc671377-0f49-4a28-93fe-0409be6a7473',
        'TCGA-AA-3970-01Z-00-DX1.712c069a-aeaf-498b-80fa-7bb481b13825',
        'TCGA-46-6026-01Z-00-DX1.e0248ea6-ae04-4960-95a8-f39b8e168a38',
        'TCGA-EJ-5505-01Z-00-DX1.8e43f020-053f-47d9-842d-4d30a776ffd2',
        'TCGA-S3-AA12-01Z-00-DX1.2A991B9F-E7E6-410B-B0BF-635E3CC40C7E',
        'TCGA-BP-4761-01Z-00-DX1.a53a55ed-a49f-4b4e-ba94-f64da0eedc4f',
        'TCGA-F5-6465-01Z-00-DX1.e0c7afc7-1904-4dc4-8e8a-83b4290f445f',
        'TCGA-CV-5977-01Z-00-DX1.DF714997-E628-476C-BD4A-CEB52FEDABD3',
        'TCGA-AG-A01L-01Z-00-DX1.1B54B139-4477-4E67-83D1-E431AA52F5B8',
        'TCGA-44-7660-01Z-00-DX1.f96c114b-f0fc-498a-97cc-e262344de357',
        'TCGA-F7-7848-01Z-00-DX1.ddeb9cb5-c474-4fdb-a9a4-7946aca5eecf',
        'TCGA-B0-5092-01Z-00-DX1.9b3a2c5c-4614-41bc-973c-601ad1f4d962',
        'TCGA-HC-A631-01Z-00-DX1.9771F112-D955-4F5A-819A-3AE84DF712C2',
        'TCGA-KK-A7AV-01Z-00-DX1.3913353D-9E72-4A3F-A627-08279D906C69',
        'TCGA-37-A5EM-01Z-00-DX1.FF7B9A1C-9D2C-43E4-9AE9-711214ACF77D',
        'TCGA-G4-6306-01Z-00-DX1.962227ca-b0d6-4cf4-afea-8f7c2f9b2477',
        'TCGA-B0-5116-01Z-00-DX1.48fbc5d2-2fda-4ad3-86ee-ab43fb0f56cb',
        'TCGA-49-4510-01Z-00-DX1.0b7c713f-6f72-42c3-967d-7e19a280e761',
        'TCGA-BP-4776-01Z-00-DX1.97ce4fa4-eaa7-47c0-8baa-5aeba076696b',
        'TCGA-A2-A0EV-01Z-00-DX1.EA8C5594-BA4F-47A8-949B-D536E00E62C9',
        'TCGA-C8-A12L-01Z-00-DX1.01863BE4-F4AB-45DE-894A-46544777C519',
        'TCGA-F7-8489-01Z-00-DX1.07122ea3-3f1a-49f8-bfb2-be532a56ae26',
        'TCGA-73-4658-01Z-00-DX1.d5beb44f-9d76-485a-8af4-407b0f1a610e',
        'TCGA-AG-3608-01Z-00-DX1.aabf7424-6c66-489d-9715-8632d9a17cfc',
        'TCGA-P3-A6T8-01Z-00-DX1.E3B6BA9A-3E96-4CCD-9C0F-52E9D27F1350',
        'TCGA-AG-3885-01Z-00-DX1.85bc7cb5-ab71-4037-85fe-df76fea4e07f',
        'TCGA-49-4490-01Z-00-DX5.feea87ff-61c6-465e-8b36-4803f6af3d62']
    #check = ['TCGA-OK-A5Q2-01Z-00-DX2.C828A160-87DF-4625-A8C5-2057F61D54F4',
             #'TCGA-AG-3885-01Z-00-DX1.85bc7cb5-ab71-4037-85fe-df76fea4e07f',
             #'TCGA-49-4506-01Z-00-DX3.a05a3969-bbef-48d8-86de-f51c8870afd6',
             #'TCGA-AG-3608-01Z-00-DX1.aabf7424-6c66-489d-9715-8632d9a17cfc',
             #'TCGA-BP-5170-01Z-00-DX1.ae43bef7-3d81-4f69-be37-b4958bf79939',
             #'TCGA-BP-4988-01Z-00-DX1.b037f17b-8fc7-46b2-adae-8e1140606fb7',
             #'TCGA-BP-4346-01Z-00-DX1.e2f7b4ce-fbe8-482e-822e-ef428731f68b',
             #'TCGA-EJ-5509-01Z-00-DX1.d5ddb1aa-3fd9-468b-abfa-97b908695a30',
             #'TCGA-CQ-7072-01Z-00-DX1.5932E8B0-B5B3-4ECC-AC61-A85FEE443536',
             #'TCGA-AR-A0TT-01Z-00-DX1.127C2A4F-62AC-4E83-99DB-13519BD2949D',
             #'TCGA-CV-5977-01Z-00-DX1.DF714997-E628-476C-BD4A-CEB52FEDABD3',
             #'TCGA-F7-A620-01Z-00-DX1.E0DA4A79-6F9B-4F7C-912E-4A4514DF8F49',
             #'TCGA-AA-3521-01Z-00-DX1.9d6be975-7be1-4f6c-99db-5101369c6624',
             #'TCGA-EJ-5505-01Z-00-DX1.8e43f020-053f-47d9-842d-4d30a776ffd2',
             #'TCGA-F7-A61S-01Z-00-DX1.BD41A8AB-B10D-4020-A00D-47A3A3827955',
             #'TCGA-49-4501-01Z-00-DX1.9c4bf212-5e32-449e-8710-cf7a3594156e',
             #'TCGA-49-4512-01Z-00-DX5.7198ce36-1fae-4da1-9f26-b7f43cf01133',
             #'TCGA-EJ-5526-01Z-00-DX1.70486c91-c1cd-4478-8c90-b103dacda0ae',
             #'TCGA-BP-4985-01Z-00-DX1.46854f1c-2f91-4990-b0ca-c76ff09bd835',
             #'TCGA-CV-A45W-01Z-00-DX1.6123D9EE-BB0B-479E-9CCF-345AAD567E9C',
             #'TCGA-AA-3667-01Z-00-DX1.28dc3612-1c43-4727-a134-698cc4315dc3',
             #'TCGA-EW-A6SD-01Z-00-DX1.32D8240E-2076-492B-BB95-300A9FCA96E7',
             #'TCGA-37-A5EM-01Z-00-DX1.FF7B9A1C-9D2C-43E4-9AE9-711214ACF77D',
             #'TCGA-KK-A6DY-01Z-00-DX1.A58E02B6-F093-43EE-B06C-975C4171E8A1',
             #'TCGA-18-3408-01Z-00-DX1.B4AFC08A-7460-4EE6-B033-629C6A6CA6E8',
             #'TCGA-AG-3586-01Z-00-DX1.38d7b4ce-9f18-48e6-9988-a3ef43b3d646',
             #'TCGA-BP-4776-01Z-00-DX1.97ce4fa4-eaa7-47c0-8baa-5aeba076696b',
             #'TCGA-EJ-5502-01Z-00-DX1.9caa4792-06a2-431d-bec3-8b17f41453fe',
             #'TCGA-CM-6676-01Z-00-DX1.dcc2bf23-ecaa-4952-8485-fc609af66298',
             #'TCGA-49-4512-01Z-00-DX7.1f758560-85e2-4cf9-a81a-317536d96bcc',
             #'TCGA-CV-6933-01Z-00-DX1.2AF4A57B-EEED-4781-8C7B-74C542D60605',
             #'TCGA-CM-6168-01Z-00-DX1.96af6eb2-9d51-4671-baf8-1a73d0c66869',
             #'TCGA-IB-AAUN-01Z-00-DX1.70F5AFF8-FECD-4BEE-AD00-E03F05D29B58',
             #'TCGA-73-4658-01Z-00-DX1.d5beb44f-9d76-485a-8af4-407b0f1a610e',
             #'TCGA-85-8288-01Z-00-DX1.0e68535f-a1f3-450c-b8fa-a859f2269da8',
             #'TCGA-BP-4998-01Z-00-DX1.f3ea98d5-3807-47d3-bcd6-5501d7679b88',
             #'TCGA-P3-A6T7-01Z-00-DX1.B150E785-D8CA-4D1A-95B7-D678E5A66AFC',
             #'TCGA-AY-A71X-01Z-00-DX1.68F9BC0F-1D60-4AEF-9083-509387038F03',
             #'TCGA-AA-3672-01Z-00-DX1.6cc142eb-e77f-4c09-a6ac-e85470221812',
             #'TCGA-KK-A8I8-01Z-00-DX1.9C138C77-7DD7-4EE0-A34C-D4B629E18087',
             #'TCGA-CV-A6JD-01Z-00-DX1.AEEF2C26-FBCB-43A9-AC34-E9C1570A1BBC',
             #'TCGA-CQ-7065-01Z-00-DX1.083ed2a3-9074-4e48-b950-7255ca54ce77',
             #'TCGA-86-7701-01Z-00-DX1.a8a6e71e-9fa9-42c6-a186-0ac7526e9960',
             #'TCGA-BP-5006-01Z-00-DX1.98cb6ac9-bb30-4b71-aa66-0d08f80ecd55',
             #'TCGA-KK-A6E8-01Z-00-DX1.5A8673FF-44A5-4383-86E8-A830205C99C7',
             #'TCGA-46-6026-01Z-00-DX1.e0248ea6-ae04-4960-95a8-f39b8e168a38',
             #'TCGA-CK-5914-01Z-00-DX1.dfa6d814-6ddb-4058-a236-d57303cbfbe9',
             #'TCGA-AG-A01L-01Z-00-DX1.1B54B139-4477-4E67-83D1-E431AA52F5B8',
             #'TCGA-BA-A6DD-01Z-00-DX1.5B67DD6F-811C-4DE2-894E-C8FE4CF7FBE7',
             #'TCGA-B6-A0WZ-01Z-00-DX1.6CFB236E-36F5-43D6-8DE3-C4ECBD3C14C6',
             #'TCGA-KU-A66S-01Z-00-DX1.195F55BC-4F8F-4039-8E12-4B18C53A8B04',
             #'TCGA-KK-A7AV-01Z-00-DX1.3913353D-9E72-4A3F-A627-08279D906C69']

    for f in wsi_paths:
        path, ext = os.path.splitext(f)
        name = os.path.basename(path)
        if name not in check:
            continue
  
        if os.path.exists(os.path.join(args.tile_path,name)) and os.listdir(os.path.join(args.tile_path,name)):
            #print('continue')
            print(f'{name} already exists and not empty. Skipping...')
            continue
        
        print('wsi_name', name)
        
        args.name = name
        print(f'Parsing {name}...')
        status = parse_wsi(args, f)
        print(f'Parsed {status}')
        if not status:
            errors.append(name)
        else:
            success["name"].append(name)
            success["time"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    err_df = pd.DataFrame({'name': errors})
    err_df.to_csv(os.path.join(args.save_path,'errors.csv'))

    succ_df = pd.DataFrame(success)
    succ_df.to_csv(os.path.join(args.save_path,'success_log.csv'))
    
    print('Finished parsing')



