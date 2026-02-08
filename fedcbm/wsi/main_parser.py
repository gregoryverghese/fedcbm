import sys
sys.path.append('/SAN/colcc/Hormad1/projects/MINOTAUR')
print(sys.path)

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

    ap.add_argument('-mz', '--map_size', default=int(209715200), type=int,
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
    print('wsipath',args.wsi_path)
    wsi_paths=glob.glob(os.path.join(args.wsi_path,'*'))
    #print(len(wsi_paths))
    #random.seed(4)
    #random.shuffle(wsi_paths)
    #random.shuffle(wsi_paths)
    #n = len(wsi_paths)
    #k = int(0.1*n)
    #index=random.sample(range(1,n),k)
    #wsi_paths = list(np.array(wsi_paths)[index])
    check = ['TCGA-BQ-5877-01Z-00-DX1.3060d47f-0bc6-4a81-82ba-bf596d790f0f']
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
    
    print('greg',len(wsi_paths))
    for f in wsi_paths:
        path, ext = os.path.splitext(f)
        name2 = os.path.basename(path)
        name = os.path.basename(path)[:16]
        if name2 not in check:
            continue
  
        if os.path.exists(os.path.join(args.tile_path,name)):
            #print('continue')
            print(f'{name} already exists and not empty. Skipping...')
            continue

        os.makedirs(os.path.join(args.tile_path,name), exist_ok=False) 
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



