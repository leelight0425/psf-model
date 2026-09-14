import argparse
from model.PSF_mlp import *
from model.optics_rgb import *
import utils.train as train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default=r'./configs/real.yaml')
    ns = parser.parse_args()
    source = ns.config
    args = train.config(source)
    seed = int(args.get('seed', 0))
    torch.manual_seed(seed)
    print(f'seed = {seed}')

    device = args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    result_path = args['result_path']
    if args.get('real', False):
        IS = IS(filepath=None, s_psf=args.get('s_psf'), sensor_res=args.get('sensor_res'),
                efl=args['efl'], pixelsize=args['pixelsize'],
                wavelengths=args['wavelengths'], na=args['na'], hfov=args['hfov'])
    else:
        IS = IS(filepath=args['in_path'], s_psf=args.get('s_psf'), sensor_res=args.get('sensor_res'),
                efl=args.get('efl'), pixelsize=args.get('pixelsize'))
    IS.seidel_basis = IS.s_basis(IS.wf_res, type=args['net'])

    net = PSF_mlp(device=device)

    shiftnet = shift_net(device=device)
    train.train(net, shiftnet, IS, args)
