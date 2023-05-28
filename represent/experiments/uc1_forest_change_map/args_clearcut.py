"""
Code Author: Ridvan Salih Kuzu, Sudipan Saha.

"""
import argparse


class ArgumentsDCVA():
    """This class defines some options required for running DCVA
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        # MANDATORY INPUTS
        parser.add_argument('--pre-s1-path',default="represent/data/UC1/clearcutting/S1/S1A_20150817_10m.tif",help='path to the pre-change image')
        parser.add_argument('--post-s1-path',default="represent/data/UC1/clearcutting/S1/S1A_20160911_10m.tif",help='path to the post-change image')
        parser.add_argument('--pre-s2-path', default="represent/data/UC1/clearcutting/S2/S2A_20150820_10m.tif",help='path to the pre-change image')
        parser.add_argument('--post-s2-path', default="represent/data/UC1/clearcutting/S2/S2A_20160913_10m.tif",help='path to the post-change image')
        parser.add_argument('--forest-mask-path', default="",help='path to the forest mask image')
        parser.add_argument('--ground-truth-path',default="represent/data/UC1/clearcutting/GT/2015-2016_magnitude_5000x5000.tif",help='path to the reference ground-truth image')
        parser.add_argument('--input-type', type=int, default=2, help=' Input Type: 1 for S1, 2 for S2, 3 for S1 + S2')

        parser.add_argument('--out-dir',default='represent/result/clearcut/',type=str, help='Out Directory (default: represent/result/)')
        parser.add_argument('--model-s1-dir',default='represent/weights/bigENTrainedSupModel_resnet18.pth',type=str, help='Model Directory (default: represent/weights)')
        #parser.add_argument('--model-s2-dir',default='represent/weights/S2_L0.0005_P0.0005_T0_1_model_best.pth',type=str, help='Model Directory (default: represent/weights)')
        parser.add_argument('--model-s2-dir',default='represent/weights/modelSupBENet4Channel.pth',type=str, help='Model Directory (default: represent/weights)')


        parser.add_argument('--ssl', action='store_true',help='If TRUE, load self-supervised model for')
        parser.add_argument('--rerun', action='store_true',help='If TRUE, calculate DCVA from temporarily generated vectors')

        # HYPERPARAMETER INPUTS
        parser.add_argument('--n-trials', type=int, default=1,help='It is the number of trials for Bayesian Hyper-parameter optimization')
        parser.add_argument('--threshold-steps', type=int, default=25,help='Number of Steps to search for sensitivity-specificity')
        parser.add_argument('--output-layers', type=int, nargs='+', default=[1], help='It can be 1, 2, 3, or 4. If multiple of them is given, the optimization function selects one of them')
        parser.add_argument('-l', '--layers-process', type=int, nargs='+', default=[5, 6, 7, 8], help='It can be 5, 6, 7, or 8. If multiple of them is given, the optimization function selects one of them')

        parser.add_argument('--object-min-size', type=int, nargs='+', default=[21],help='Minimum size for objects in pixel')
        parser.add_argument('--morphology-size', type=int, nargs='+', default=[11],help='Morphological operation size for image post-processing')
        parser.add_argument('--is-saturate', type=bool, default=True,help='BOOLEAN indicating if images are preprocessed to saturate high values')
        parser.add_argument('--top-saturate', type=float, nargs='+', default=[2.0], help='Percentage to saturate')
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
