"""Wrapper to train and test a video classification model."""
from slowfast.config.my_defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args

import my_test_net



def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    test = my_test_net.test
    
    print(args)

    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
        # Perform multi-clip testing.
        #if cfg.TEST.ENABLE: #Sta roba va tutta hard-coded, da linea di comando non se la prende
        test(cfg)


if __name__ == "__main__":
    main()
