from argparse import ArgumentParser
from functools import partial

from onmt.model_builder import build_model

from onmt.utils.logging import init_logger, logger

from onmt.train_single import configure_process, _get_model_opts

from onmt.inputters.fields import load_fields
from onmt.models.model_saver import load_checkpoint
from onmt.opts import train_opts


def pretrain(opt, fields, transforms_cls, checkpoint, device_id,
         batch_queue=None, semaphore=None):
    """
    :param opt:
    :param fields:
    :param transforms_cls:
    :param checkpoint:
    :param device_id:
    :param batch_queue:
    :param semaphore:
    :return:
    """
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)

    model_opt = _get_model_opts(opt, checkpoint=checkpoint)

    # load pretrain model.
    pretrained_model = build_model(model_opt, opt, fields, checkpoint)
    pretrained_model.count_parameters(log=logger.info)




def finetune(opt):
    pretrained_model=load_checkpoint(ckpt_path=opt.pretrain_from)#TODO 定义opt.pretrain_from
    fields=load_fields(opt.save_data, pretrained_model)
    train_process = partial(
        single_pretrain,
        fields=fields,
        transforms_cls=None,
        checkpoint=pretrained_model)

def _get_parser():
    parser = ArgumentParser(description='train.py')
    train_opts(parser)
    return parser


def main():
    #从命令行调用
    parser = _get_parser()
    opt, unknown = parser.parse_known_args()
    finetune(opt)


if __name__ == "__main__":
    main()