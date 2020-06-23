import h5py
import matplotlib.pyplot as plt
filename = "test_catvnoncat.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    for key in f.keys():
        print("keys", key)
    group1=f["test_set_x"];
    group1_img=group1.value
    group2=f["test_set_y"];
    group2_img_tag=group2.value
    print("value:Key", group1_img,group2_img_tag)


#Show One image Value
plt.imshow(group1_img[0])
plt.show()