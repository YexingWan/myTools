import os
import sys
import argparse
sys.path.append("../")
sys.path.append("../../models_tensorflow/models/research/slim")
from nets import nets_factory
from restore import tf_basic_restore


def main(args):
    net_fn = nets_factory.get_network_fn(name=args.model_name,
                                         num_classes=args.num_classes,
                                         is_training=False)
    tf_basic_restore(net_fn,
                     model_name=args.model_name,
                     checkpoint_path=args.checkpoint_path,
                     input_shape=tuple(args.input_shape),
                     from_ema=args.from_ema,
                     ignore_missing_vars=args.ignore_missing_vars,
                     restore_dir=args.restore_path)


parse = argparse.ArgumentParser()
parse.add_argument("--model_name",
                   type=str,
                   choices=list(nets_factory.networks_map.keys()),
                   default="resnet_v1_50")

parse.add_argument("--num_classes",
                   type=int,
                   default=1000)


parse.add_argument("--checkpoint_path",
                   type=str,
                   required=True)

parse.add_argument("--input_shape",
                   nargs=2,
                   type=int,
                   default=[224,224])

parse.add_argument("--ignore_missing_vars",
                   action="store_true")

parse.add_argument("--from_ema",
                   action="store_true")

parse.add_argument("--restore_path",
                   type=str,default="./restored")


args = parse.parse_args()

if os.path.exists(args.checkpoint_path):
    if os.path.isdir(args.checkpoint_path):
        assert (os.path.isfile(os.path.join(args.checkpoint_path,"checkpoint")), "{} not fountd.".format(os.path.join(args.checkpoint_path,"checkpoint")))
else:
    raise FileNotFoundError("Checkpoint not found.")

if __name__ == "__main__":
    main(args)





