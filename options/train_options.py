from .base_options import BaseOptions
from typing import Union


def validate_epoch_load(epoch_load: Union[str, dict, int]) -> dict:
    if isinstance(epoch_load, str):
        if epoch_load.lower() == 'latest':
            return {'G0': 'latest', 'G1': 'latest', 'G2': 'latest', 'G3': 'latest', 'G4': 'latest', 'G5': 'latest',
                    'G6': 'latest',
                    'D0': 'latest', 'D1': 'latest', 'D2': 'latest', 'S0': 'latest', 'S1': 'latest', 'S2': 'latest'}
        elif epoch_load.lower() == 'lastest':
            return {'G0': 'latest', 'G1': 'latest', 'G2': 'latest', 'G3': 'latest', 'G4': 'latest', 'G5': 'latest',
                    'G6': 'latest',
                    'D0': 'latest', 'D1': 'latest', 'D2': 'latest', 'S0': 'latest', 'S1': 'latest', 'S2': 'latest'}
        elif epoch_load.isdigit():
            return {'G0': int(epoch_load), 'G1': int(epoch_load), 'G2': int(epoch_load), 'G3': int(epoch_load),
                    'G4': int(epoch_load), 'G5': int(epoch_load), 'G6': int(epoch_load),
                    'D0': int(epoch_load), 'D1': int(epoch_load), 'D2': int(epoch_load), 'S0': int(epoch_load),
                    'S1': int(epoch_load), 'S2': int(epoch_load)}
        else:
            raise ValueError("epoch_load must be 'latest', or an integer as a string.")
    elif isinstance(epoch_load, int):
        return {'G0': int(epoch_load), 'G1': int(epoch_load), 'G2': int(epoch_load), 'G3': int(epoch_load),
                'G4': int(epoch_load), 'G5': int(epoch_load), 'G6': int(epoch_load),
                'D0': int(epoch_load), 'D1': int(epoch_load), 'D2': int(epoch_load), 'S0': int(epoch_load),
                'S1': int(epoch_load), 'S2': int(epoch_load)}
    elif isinstance(epoch_load, dict):
        ret = {'G0': -1, 'G1': -1, 'G2': -1, 'G3': -1, 'G4': -1, 'G5': -1, 'G6': -1, 'D0': -1, 'D1': -1, 'D2': -1,
               'S0': -1, 'S1': -1, 'S2': -1}
        if 'G' in epoch_load:
            G = int(epoch_load['G']) if not isinstance(epoch_load['G'], str) else epoch_load['G']
            ret.update({'G0': G, 'G1': G, 'G2': G, 'G3': G, 'G4': G, 'G5': G, 'G6': G})
        if 'D' in epoch_load:
            D = int(epoch_load['D']) if not isinstance(epoch_load['D'], str) else epoch_load['D']
            ret.update({'D0': D, 'D1': D, 'D2': D})
        if 'S' in epoch_load:
            S = int(epoch_load['S']) if not isinstance(epoch_load['S'], str) else epoch_load['S']
            ret.update({'S0': S, 'S1': S, 'S2': S})
        for key in epoch_load:
            if key in list(ret.keys()):
                ret[key] = epoch_load[key] if isinstance(epoch_load[key], str) else int(epoch_load[key])
        return ret


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True

        self.parser.add_argument('--continue_train', type=bool, default=True,
                                 help='continue training: load the latest model')
        self.parser.add_argument('--simple_train', type=bool, default=False,
                                 help='continue training: load the latest model')
        self.parser.add_argument('--simple_train_channel', type=int, default=0,
                                 help='alternate between 0/1 and 0/2 mod every n steps')
        self.parser.add_argument('--which_epoch', type=int, default=40,
                                 help='which epoch to load if continuing training')
        self.parser.add_argument('--epoch_load', type=validate_epoch_load, default='latest', #validate_epoch_load({'G': 'latest', 'S': 100, 'S2': 'latest', 'D': 'latest'}),  #'latest', #validate_epoch_load(
                                 # {'G0': 'latest', 'G1': -1, 'G2': 'latest', 'G3': 'latest', 'G4': -1, 'G5': 'latest', 'G6': 'latest',
                                 #  'D0': 'latest', 'D2': 'latest', 'S': 'latest'}),
                                 help='which epoch to load if continuing training')
        self.parser.add_argument("--partial_train", type=Union[dict, None],
                                 default={'G': [0, 1, 2, 3, 4, 5, 6], 'D': [0, 1, 2], 'S': [2]},
                                 help="Which domains of G - D - S are trained in ["
                                      "0 - 3: visible Encoder - Decoder"
                                      "1 - 4: IR Encoder - Decoder"
                                      "2 - 5: Vis_night Encoder - Decoder"
                                      "6: Hybrid IR-Vis Night Fusion pipe - Only for G"
                                      "None: all].")
        self.parser.add_argument('--phase', type=str, default='train',
                                 help='train, val, test, etc (determines name of folder to load from)')
        self.parser.add_argument('--IR_edge_path', type=str, default='FLIR_datasets/FLIR_IR_edge_map/',
                                 help='the folder to load IR image edge map')
        self.parser.add_argument('--Vis_edge_path', type=str, default='FLIR_datasets/FLIR_Vis_edge_map/',
                                 help='the folder to load Visible image edge map')
        self.parser.add_argument('--Vis_mask_path', type=str, default='FLIR_datasets/FLIR_Vis_seg_mask/',
                                 help='the folder to load Visible image segmentation mask')
        self.parser.add_argument('--IR_mask_path', type=str, default='FLIR_datasets/FLIR_IR_seg_mask/',
                                 help='the folder to load NTIR image segmentation mask')
        self.parser.add_argument('--IR_FG_txt', type=str, default='FLIR_txt_file/IR_FG_list.txt',
                                 help='the txt file of IR image contains large FG region')
        self.parser.add_argument('--Vis_FG_txt', type=str, default='FLIR_txt_file/Vis_FG_list.txt',
                                 help='the txt file of Visible image contains large FG region')
        self.parser.add_argument('--FB_Sample_Vis_txt', type=str, default='FLIR_txt_file/FB_Sample_Vis.txt',
                                 help='txt file indicating whether feedback modulation is applied to the visible image.')
        self.parser.add_argument('--FB_Sample_IR_txt', type=str, default='FLIR_txt_file/FB_Sample_IR.txt',
                                 help='txt file indicating whether feedback modulation is applied to the NTIR image.')
        self.parser.add_argument('--IR_patch_classratio_txt', type=str, default='FLIR_txt_file/IR_patch_classratio.txt',
                                 help='the txt file of indicating the percentage of each category in the NTIR image.')

        self.parser.add_argument('--ssim_winsize', type=int, default=11, help='window size of SSIM loss')
        self.parser.add_argument('--encoded_nc', type=int, default=128, help='channel number of encoded tensor')

        self.parser.add_argument('--niter', type=int, default=150,
                                 help='# of epochs at starting learning rate (try 50*n_domains)')
        self.parser.add_argument('--niter_decay', type=int, default=150,
                                 help='# of epochs to linearly decay learning rate to zero (try 50*n_domains)')

        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for ADAM')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')

        self.parser.add_argument('--lambda_cycle', type=float, default=5.0,
                                 help='weight for cycle loss (A -> B -> A)')  #5
        self.parser.add_argument('--lambda_identity', type=float, default=1.0,  # 1.0
                                 help='weight for identity "autoencode" mapping (A -> A)')
        self.parser.add_argument('--lambda_latent', type=float, default=0.5,  # 0.5
                                 help='weight for latent-space loss (A -> z -> B -> z)')
        self.parser.add_argument('--lambda_forward', type=float, default=0.2,
                                 help='weight for forward loss (A -> B; try 0.2)')
        self.parser.add_argument('--lambda_ssim', type=float, default=2.0, help='weight for SSIM loss')
        self.parser.add_argument('--lambda_tv', type=float, default=5.0, help='weight for TV loss')  # 5.
        self.parser.add_argument('--lambda_sc', type=float, default=1.0, help='weight for SC loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=1.0, help='weight for VGG loss')
        self.parser.add_argument('--vis_prob_th', type=float, default=0.925,
                                 help='probability threshold for updating visible image segmentation GT')
        self.parser.add_argument('--vis_night_hl_th', type=float, default=0.9,
                                 help='probability threshold for updating visible image segmentation GT')
        self.parser.add_argument('--IR_prob_th', type=float, default=0.9,
                                 help='probability threshold for updating IR image segmentation GT')
        self.parser.add_argument('--grad_th_vis', type=float, default=0.8,
                                 help='threshold for SGA Loss about gradient of fake IR image')
        self.parser.add_argument('--grad_th_IR', type=float, default=0.8,
                                 help='threshold for SGA Loss about gradient of fake Visible image')
        self.parser.add_argument('--lambda_color', type=float, default=0.05,
                                 help='weight for color consistence loss')
        self.parser.add_argument('--lambda_sga', type=float, default=0.5,
                                 help='weight for gradient orientation consistence loss')

        # Epoch schedule
        self.parser.add_argument('--SGA_start_epoch', type=int, default=0,
                                 help='# of epochs at starting gradient orientation consistence loss')
        self.parser.add_argument('--SGA_fullload_epoch', type=int, default=0,
                                 help='# of epochs at starting gradient orientation consistence loss with full loaded weights')
        self.parser.add_argument('--SSIM_start_epoch', type=int, default=10, help='# of epochs at starting SSIM loss')
        self.parser.add_argument('--SSIM_fullload_epoch', type=int, default=10,
                                 help='# of epochs at starting SSIM loss with full loaded weights')
        self.parser.add_argument('--netS_start_epoch', type=int, default=20,
                                 help='# of epochs at starting semantic consistency loss')
        self.parser.add_argument('--netS_end_epoch', type=int, default=75,
                                 help='# of epochs at stopping weights update in netS')
        self.parser.add_argument('--updateGT_start_epoch', type=int, default=30,
                                 help='# of epochs at starting updating uncertain region of segmentation GT')
        self.parser.add_argument("--often_balance", action='store_true', help="balance the appearance times.")
        self.parser.add_argument("--max_value", type=float, default=7, help="Max Value of Class Weight.")
        self.parser.add_argument("--only_hard_label", type=float, default=0, help="class balance.")
        self.parser.add_argument("--lambda_CGR", type=float, default=1.0,
                                 help="weight for the conditional gradient repair loss.")
        self.parser.add_argument("--sqrt_patch_num", type=int, default=8,
                                 help="sqrt patch number for structral gradient alignment loss.")
        self.parser.add_argument("--partial_train_stop", type=int, default=100,
                                 help="Epoch to stop partial training of G.")

        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_step_latest', type=int, default=500,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--display_freq', type=int, default=200,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=200,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--pool_size', type=int, default=10,
                                 help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
