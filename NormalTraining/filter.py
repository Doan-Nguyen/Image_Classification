import os 
import shutil
list_dir = next(os.walk("example_dataset_structure"))[1]
for author in list_dir:
	author_path = os.path.join("example_dataset_structure", author)
	list_img = next(os.walk(author_path))[2]
	for img_name in list_img:
		img_path = os.path.join(author_path,img_name)
		os.remove(img_path)
	shutil.copyfile("./image_0004.jpg", os.path.join(author_path, "image_0004.jpg"))
