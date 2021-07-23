"""
validate a model
why isolate validting process from training: limit of GPU memory
"""
from onmt.inputters.fields import load_fields
from onmt.models.model_saver import load_checkpoint
from onmt.opts import train_opts
from onmt.transforms import get_transforms_cls
from onmt.utils import set_random_seed
from onmt.utils.logging import init_logger, logger
from onmt.train_single import validate as single_val
from onmt.utils.parse import ArgumentParser
from functools import partial


def _get_parser():
    parser = ArgumentParser(description='validate.py')
    train_opts(parser)
    return parser


def _init_validate(opt):
    "加载要验证的模型"

    checkpoint = load_checkpoint(ckpt_path=opt.validate_from)
    fields=load_fields(opt.save_data,checkpoint)
    transforms_cls = get_transforms_cls(opt._all_transform)
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))
    return checkpoint, fields, transforms_cls


def validate(opt):
    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)

    #加载需验证的模型及验证数据的格式
    Model, fields, transforms_cls = _init_validate(opt)
    valid_process=partial(single_val,fields=fields,transforms_cls=transforms_cls,checkpoint=Model)
    valid_process(opt,device_id=0)


def main():
    #从命令行调用
    parser = _get_parser()
    opt, unknown = parser.parse_known_args()
    validate(opt)


if __name__ == "__main__":
    main()