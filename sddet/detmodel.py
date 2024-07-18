from engine.model.base_model import BaseModel
from engine.model.build import BUILD_MODEL_REGISTRY


@BUILD_MODEL_REGISTRY.register()
class DetModel(BaseModel):
    def __init__(self, cfg):
        super(DetModel, self).__init__(cfg)
        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        此方法必须实现
        """
        img = data['img'].to(self.device, non_blocking=True)
        gt_bboxes = [bbox.to(self.device, non_blocking=True) for bbox in data['gt_bboxes']]
        gt_masks =  [mask.to(self.device, non_blocking=True) for mask in data['gt_masks']]
        gt_labels = [label.to(self.device, non_blocking=True) for label in data['gt_labels']]
        return self.g_model.forward_train(img, gt_labels, gt_masks, gt_bboxes)

    def generator(self, data):
        """
        此方法必须实现
        """
        img = data['img'].to(self.device, non_blocking=True)
        result = self.g_model(img)
        return result
