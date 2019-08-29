#!/usr/bin/env python

"""
python.exe ./batch-crop.py "../us/persp/imgs/*.png"
"""

import glob
import os
import yaml
import sys

directory = './batch'


def batch(ins, config, out_dir):
    for key, conf in config['crops'].items():
        batch_crop(ins, key, conf, out_dir)


def batch_crop(ins, name, config, out_dir):
    print("doing batch %s" % name)
    for f in ins:
        base = os.path.basename(f)
        sp = os.path.splitext(base)
        new_base = "%s_%s_%s" % (sp[0], name, sp[1])
        new_name = "%s/%s" %(out_dir, new_base)
        cmd = "C:/apps/ImageMagick/magick.exe %s -crop %s %s" % (f, config, new_name)
        # print(cmd)
        os.system(cmd)


imgs = glob.glob(sys.argv[1])
assert len(imgs) > 0, "Must have at least one image to convert."

if not os.path.exists(directory):
    os.makedirs(directory)

with open("crops.yaml", 'r') as stream:
    try:
        batch(imgs, yaml.safe_load(stream), directory)
    except yaml.YAMLError as exc:
        sys.exit(exc)
