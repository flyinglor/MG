import os
import shutil

class models_genesis_config:
    model = "Unet3D"
    suffix = "MG"
    exp_name = model + "-" + suffix
    
    # data
    # data = "/mnt/dataset/shared/zongwei/LUNA16/Self_Learning_Cubes"
    # train_fold=[0,1,2,3,4]
    # valid_fold=[5,6]
    # test_fold=[7,8,9]
    data = "/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/UKB_CAT12/generated_cubes"
    train_fold=[5,6,7]
    valid_fold=[8]
    test_fold=[9]
    hu_min = 0.0
    hu_max = 4000.0
    # hu_min = -1000.0
    # hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 8
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    pretrain_epoch = 100
    nb_epoch = 100
    patience = 100
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Hui/MG_ukb"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    disable_wandb = True
    dataset = "DZNE"
    # dataset = "HOSPITAL"
    # pretrain_dsname = "UKB"
    pretrain_dsname = "ADNI"
    checkpoint_dir = 'pretrained_weights/Genesis_ADNI_3_2.pt'
    logs_path = "logs/"
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
