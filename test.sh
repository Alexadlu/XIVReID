#!/usr/bin/env bash
python tools/test-sysu.py --config_file='configs/all.yml' MODEL.DEVICE_ID "('9')" TEST.WEIGHT "('./logs/sysu/cosine/resnet50_model_120.pth')"