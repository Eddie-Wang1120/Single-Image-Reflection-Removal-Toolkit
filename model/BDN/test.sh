#!/usr/bin/env bash
python3 ./test.py --dataroot ../dataset \
    --batchSize 1 \
    --norm batch \
    --which_model_netG cascade_unet \
    --ns 7,5,5 \
    --iteration 1 \
    --outf /home/wjh/Desktop/SIRR_Toolkit/model/output/BDN \
    --netG ./model/model.pth \
    
