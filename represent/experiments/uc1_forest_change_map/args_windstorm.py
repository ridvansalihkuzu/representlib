"""
Code Author: Ridvan Salih Kuzu, Sudipan Saha.

"""
import argparse
import os

class ArgumentsDCVA():
    """This class defines some options required for running DCVA
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # MANDATORY INPUTS
        parser.add_argument('--pre-s1-path',default="/local_home/kuzu_ri/GIT_REPO/representlib/represent/data/UC1/windstorm_damage/S1/S1B_IW_GRDH_1SDV_20210601.tif",help='path to the pre-change image')
        parser.add_argument('--post-s1-path',default="/local_home/kuzu_ri/GIT_REPO/representlib/represent/data/UC1/windstorm_damage/S1/S1B_IW_GRDH_1SDV_20210707.tif",help='path to the post-change image')
        parser.add_argument('--pre-s2-path', default="/local_home/kuzu_ri/GIT_REPO/representlib/represent/data/UC1/windstorm_damage/S2/S2B_MSIL2A_20210604.tif",help='path to the pre-change image')
        parser.add_argument('--post-s2-path', default="/local_home/kuzu_ri/GIT_REPO/representlib/represent/data/UC1/windstorm_damage/S2/S2B_MSIL2A_20210628.tif",help='path to the post-change image')
        parser.add_argument('--forest-mask-path', default="/local_home/kuzu_ri/GIT_REPO/representlib/represent/data/UC1/windstorm_damage/GT/forest_mask.tif",help='path to the forest mask image')
        parser.add_argument('--ground-truth-intact-path',default="/local_home/kuzu_ri/GIT_REPO/representlib/represent/data/UC1/windstorm_damage/GT/intact_stands_area.tif",help='path to the reference ground-truth image')
        parser.add_argument('--ground-truth-disturbance-path',default="/local_home/kuzu_ri/GIT_REPO/representlib/represent/data/UC1/windstorm_damage/GT/damaged_stands_extended_set_area.tif",help='path to the reference ground-truth image')
        parser.add_argument('--input-type', type=int, default=2, help=' Input Type: 1 for S1, 2 for S2, 3 for S1 + S2')

        parser.add_argument('--out-dir',default='/local_home/kuzu_ri/GIT_REPO/representlib/represent/result/windstorm/',type=str, help='Out Directory (default: represent/result/)')
        #parser.add_argument('--model-dir', default='/local_home/kuzu_ri/GIT_REPO/representlib/represent/weights', type=str,help='Model Directory (default: represent/weights)')
        parser.add_argument('--model-s1-dir',default='represent/weights/bigENTrainedSupModel_resnet18.pth',type=str, help='Model Directory (default: represent/weights)')
        #parser.add_argument('--model-s1-dir',default='/local_home/kuzu_ri/GIT_REPO/representlib/represent/weights/S1_L0.001_P5e-06_T0.5_D0.9_P0_N0_15_model_best.pth',type=str, help='Model Directory (default: represent/weights)')
        #parser.add_argument('--model-s2-dir',default='/local_home/kuzu_ri/GIT_REPO/representlib/represent/weights/S2_L0.0005_P5e-06_T0.1_PTrue_N0_5_model_best.pth',type=str, help='Model Directory (default: represent/weights)')
        parser.add_argument('--model-s2-dir',default='/local_home/kuzu_ri/GIT_REPO/representlib/represent/weights/modelSupBENet4Channel.pth',type=str, help='Model Directory (default: represent/weights)')
        parser.add_argument('--ssl', action='store_true',help='If TRUE, load self-supervised model for')
        parser.add_argument('--rerun', action='store_true',help='If TRUE, calculate DCVA from temporarily generated vectors')

        # HYPERPARAMETER INPUTS
        parser.add_argument('--n-trials', type=int, default=1,help='It is the number of trials for Bayesian Hyper-parameter optimization')
        parser.add_argument('--threshold-steps', type=int, default=25,help='Number of Steps to search for sensitivity-specificity')
        parser.add_argument('--output-layers', type=int, nargs='+', default=[1], help='It can be 1, 2, 3, or 4. If multiple of them is given, the optimization function selects one of them')
        parser.add_argument('-l', '--layers-process', type=int, nargs='+', default=[5, 6, 7, 8], help='It can be 5, 6, 7, or 8. If multiple of them is given, the optimization function selects one of them')

        parser.add_argument('--object-min-size', type=int, nargs='+', default=[7,19,23],help='Minimum size for objects in pixel')
        parser.add_argument('--morphology-size', type=int, nargs='+', default=[17,23],help='Morphological operation size for image post-processing')
        parser.add_argument('--is-saturate', type=bool, default=True,help='BOOLEAN indicating if images are preprocessed to saturate high values')
        parser.add_argument('--top-saturate', type=float, nargs='+',  default=[1.0,2.0], help='Percentage to saturate')
        #parser.add_argument('-e', '--end-step', type=int, default=8)
        # parser.add_argument('--output-start', type=int, default=1,help='comma separated layers from which features are extracted')

        self.initialized = True
        return parser

    def parseOptions(self):
        """Parse the options"""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args()
        return opt
