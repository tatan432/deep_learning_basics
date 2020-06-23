#Importing Dataset
import h5py
filename_test = "test_catvnoncat.h5"
filename_train = "train_catvnoncat.h5"


def load_image():

    with h5py.File(filename_test, "r") as f:
        # List all groups
        for key in f.keys():
            print("keys", key)
        group1=f["test_set_x"];
        test_img_x=group1[()]
        group2=f["test_set_y"];
        test_img_y=group2[()]
    
    with h5py.File(filename_train, "r") as f:
        # List all groups
        for key in f.keys():
            print("keys", key)
        group1=f["train_set_x"];
        train_img_x=group1[()]
        group2=f["train_set_y"];
        train_img_y=group2[()]
        
    return train_img_x, train_img_y, test_img_x, test_img_y


