from datetime import datetime
import os

def get_name_from_time(folder_to_write):
    dt = datetime.now()
    year, month, day, hour, minute, second, microsecond = dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
    write_time = str(year) + "_" + str(month) + "_" + str(day) + \
                    "_" + str(hour) + "_" + str(minute) + "_" + \
                     str(second) + "_" + str(microsecond)
    img_path = os.path.join(folder_to_write, 'image_'+ write_time +'.jpg')
    return img_path

def get_url(ori_url):
    result = ""
    s3_index = ori_url.find("amazonaws.com")
    list_img_extention = ['.jpg', '.png', '.JPG', '.PNG', 'jpeg', 'JPEG']
    for img_extention in list_img_extention:
        ext_index = ori_url.find(img_extention)
        if ext_index != -1:
            result = ori_url[s3_index + 14 :ext_index+4]
            return result