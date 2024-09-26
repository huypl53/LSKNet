import asyncio
import json
import os
from typing import List, Tuple

from mmdet.apis import init_detector
from sqlalchemy import select
from mmrotate.apis import inference_detector_by_patches

from app.connection import ftpTransfer
from app.db.connector import get_db
from app.model.task import TaskMd
from app.schema import EnhancementOutput, EnhancementParam, DetectShipParam
from app.service.binio import read_ftp_image, write_ftp_image
from enhancing.core import adjust_gamma, hist_equalize
from log import logger


async def async_main():
    while True:
        a_session = anext(get_db())
        session = await (a_session)
        stmt = (
            select(TaskMd)
            .where(TaskMd.task_type == 5)
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        results = await session.execute(stmt)
        mapping_results = results.mappings().all()
        tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]

        params: DetectShipParam = None
        model = None
        print("----------")
        try:
            for i, t in enumerate(tasks):
                if i == 1:
                    break  # update only one
                param_dict = json.loads(t.task_param)
                if not t.task_param:
                    params = DetectShipParam()
                elif ((not params) or t.task_param != params.model_dump()):
                    try:
                        params = DetectShipParam(**param_dict)
                    except Exception as e:
                        t.task_stat = 0
                        t.task_message = str(e)
                else:
                    t.task_stat = 0  # task got error
                    t.task_message = 'Init model failed!'
                    continue
                model = init_detector(
                    params.config, params.checkpoint, device=params.device)

                im = read_ftp_image(params.input_file)
                result = inference_detector_by_patches(model, im, params.patch_sizes,
                                                       params.patch_steps, params.img_ratios,
                                                       params.merge_iou_thr)  # inference for batch
                if not len(result):
                    t.task_stat = 1
                    t.task_message = 'No detection'
                    continue
                output = result[0]
                t.task_output = json.dumps(output)
                t.task_stat = 1
                t.task_message = 'Successful'
                t.task_param = json.dumps(params.model_dump())
                t.process_id = os.getpid()
        except Exception as e:
            logger.error(e)

        print("----------")
        await session.commit()
        await session.close()
        await asyncio.sleep(3)


# Run this from outter directory
# python ./lsknet/huge_images_extract.py --dir ./images  --config './lsknet/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py' --checkpoint './epoch_3_050324.pth' --score-thr 0.5 --save-dir /tmp/ships/
