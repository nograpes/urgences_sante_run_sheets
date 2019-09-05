from PIL import Image
import os


def multiplicative_inverse_blend(acc, item, index):
    """
    Correct "running" average image valid for less than 255 images (since one channel)
    """
    return Image.blend(acc, item, alpha=1 / (index + 1))


class Compositor:
    def __init__(self, func):
        self.num = 0
        self.func = func
        self.composite = None  # Make white image

    def __repr__(self):
        return "compositor of %s images" % self.num

    @staticmethod
    def get_running_avg_compositor():
        return Compositor(multiplicative_inverse_blend)

    def compose(self, img_dat):
        """Compose additional image"""
        img = Image.fromarray(img_dat)
        # The first composite is always the input image
        if self.composite is None:
            self.composite = img
            return self.composite
        # Change the composite builder
        self.composite = self.func(self.composite, img, self.num)
        # Index of the current composition
        self.num = self.num + 1

    def save(self, filename):
        """Save composite image"""

        split = os.path.splitext(filename)
        base = split[0]
        ext = split[1]

        composites = self.composite if isinstance(self.composite, list) else [self.composite]

        i = 0
        for composite in composites:
            composite.save("%s_%s%s" % (base, i, ext))
            i = i + 1
