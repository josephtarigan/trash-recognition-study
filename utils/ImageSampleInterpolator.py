'''
This will tranform image in the given folder into several set of image
This module does the things as introduced in Inception paper where the researcher did multiscale training
The input image will be resized into 4 scale
256, 288, 320, and 352
It will resize the image's smallest axis size into the specified length
For each scaled image, take 224 x 224 portion of each corner, centre, and a resized 224 x 224 of the image
Finally, create the mirrored image out of those images
'''

import argparse
import os
import logging
from PIL import Image
from fs import open_fs
from fs.walk import Walker

parser = argparse.ArgumentParser('Specify the dataset directory')
parser.add_argument('--src', '--source', required=True, help='Specify the source directory')
parser.add_argument('--dest', '--destination', required=True, help='Specify the destination directory')
args = vars(parser.parse_args())

src_folder = args['src']
dest_folder = args['dest']

target_width = 244
target_height = 244


def get_ratio (image, target):
    ratio = 0
    if (image.height < image.width):
        ratio = target/image.height
    elif (image.width < image.height):
        ratio = target/image.width
    else:
        ratio = target/image.height
    return ratio

def resize_image (image, target_ratio):
    resized_image = image.resize((int(image.height * target_ratio), int(image.width * target_ratio)), Image.ANTIALIAS)
    return resized_image

def derive_samples (image, target_height, target_width):
    left_top = image.crop((0, 0, target_width, target_height))
    right_top = image.crop((image.width-target_width, 0, image.width, target_height))
    left_bottom = image.crop((0, image.height-target_height, target_width, image.height))
    right_bottom = image.crop((image.width-target_width, image.height-target_height, image.width, image.height))
    centre = image.crop((
                            image.width/2 - target_width/2
                            , image.height/2 - target_height/2
                            , image.width/2 + target_width/2
                            , image.height/2 + target_height/2)
                        )
    resized = image.resize((target_height, target_width), Image.ANTIALIAS)

    mirrored_left_top = left_top.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_right_top = right_top.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_left_bottom = left_bottom.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_right_bottom = right_bottom.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_centre = centre.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_resized = resized.transpose(Image.FLIP_LEFT_RIGHT)

    return left_top, right_top, left_bottom, right_bottom, centre, resized, mirrored_left_top, mirrored_right_top, mirrored_left_bottom,  mirrored_right_bottom, mirrored_centre, mirrored_resized


def save_samples (lt, rt, lb, rb, ce, re, mlt, mrt, mlb, mrb, mce, mre, dest_path, file_name):
    logging.info("Writing LT to " + os.path.join(dest_path, file_name, "LT", ".jpg"))
    lt.save(dest_path + os.sep + file_name + "_LT" + ".jpg", "JPEG")
    
    logging.info("Writing RT to " + os.path.join(dest_path, file_name, "RT", ".jpg"))
    rt.save(dest_path + os.sep + file_name + "_RT" + ".jpg", "JPEG")
    
    logging.info("Writing LB to " + os.path.join(dest_path, file_name, "LB", ".jpg"))
    lb.save(dest_path + os.sep + file_name + "_LB" + ".jpg", "JPEG")

    logging.info("Writing RB to " + os.path.join(dest_path, file_name, "RB", ".jpg"))
    rb.save(dest_path + os.sep + file_name + "_RB" + ".jpg", "JPEG")
    
    logging.info("Writing CE to " + os.path.join(dest_path, file_name, "CE", ".jpg"))
    ce.save(dest_path + os.sep + file_name + "_CE" + ".jpg", "JPEG")

    logging.info("Writing RE to " + os.path.join(dest_path, file_name, "RE", ".jpg"))
    re.save(dest_path + os.sep + file_name + "_RE" + ".jpg", "JPEG")

    logging.info("Writing MLT to " + os.path.join(dest_path, file_name, "MLT", ".jpg"))
    mlt.save(dest_path + os.sep + file_name + "_MLT" + ".jpg", "JPEG")

    logging.info("Writing MRT to " + os.path.join(dest_path, file_name, "MRT", ".jpg"))
    mrt.save(dest_path + os.sep + file_name + "_MRT" + ".jpg", "JPEG")
    
    logging.info("Writing MLB to " + os.path.join(dest_path, file_name, "MLB", ".jpg"))
    mlb.save(dest_path + os.sep + file_name + "_MLB" + ".jpg", "JPEG")
    
    logging.info("Writing MRB to " + os.path.join(dest_path, file_name, "MRB", ".jpg"))
    mrb.save(dest_path + os.sep + file_name + "_MRB" + ".jpg", "JPEG")
    
    logging.info("Writing MCE to " + os.path.join(dest_path, file_name, "MCE", ".jpg"))
    mce.save(dest_path + os.sep + file_name + "_MCE" + ".jpg", "JPEG")
    
    logging.info("Writing MRE to " + os.path.join(dest_path, file_name, "MRE", ".jpg"))
    mre.save(dest_path + os.sep + file_name + "_MRE" + ".jpg", "JPEG")
    return None


def process_image (image_path, dest_folder, target_height, target_width):
    image = Image.open(image_path)

    size = 244
    resized = resize_image(image, get_ratio(image, size))
    # get category
    splited_path = os.path.dirname(image_path).split(os.sep)
    category = splited_path[len(splited_path)-1]
    # create result folder
    result_dest_folder = os.path.join(dest_folder, category)
    if (not os.path.exists(result_dest_folder)):
        os.makedirs(result_dest_folder)
    # resulting file name
    file_name = (image_path).split(os.sep)[len((image_path).split(os.sep))-1].split('.')[0]
    # shape the image
    lt, rt, lb, rb, ce, re, mlt, mrt, mlb, mrb, mce, mre = derive_samples(resized, target_height, target_width)
    # save the images
    save_samples(lt, rt, lb, rb, ce, re, mlt, mrt, mlb, mrb, mce, mre, result_dest_folder, file_name)


def folder_walker(src_folder, dest_folder, target_height, target_width):
    src_fs = open_fs(src_folder)
    walker = Walker(filter=['*.jpg'])
    for path in walker.files(src_fs):
        image_path = os.path.normpath(src_folder + os.sep + path)
        process_image(image_path, dest_folder, target_height, target_width)
    return None


folder_walker(src_folder, dest_folder, target_height, target_width)