"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ..preprocessing.generator import Generator
from ..utils.image import read_image_bgr

import os
import numpy as np
from six import raise_from
from PIL import Image

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

voc_classes = {
    'company name'				:1,
    'company logo'				:2,
    'close button'				:3,
    'type username'				:4,
    'type password'				:5,
    'login/sign in heading'			:6,
    'sign in button'				:7,
    'password forgotten link'			:8,
    'username forgotten link'			:9,
    'link to another_page = link to other page' :10,
    'login with facebook = connect with facebook':11,
    'login with twitter = connect with twitter'	:12,
    'login with google = connect with google'	:13,
    'tick box to remember'			:14,
    'new registration'				:15,
    'help'					:16,
    'dropdown'					:17,
    'profile picture'				:18,
    'search field'				:19,
    'next button'				:20,
    'first name'				:21,
    'last name'					:22,
    'login with linkedIn = connect with linkidin':23,
    'type code'					:24,
    'cancel button'				:25,
    'tick box if necessary'			:26,
    'login with github'				:27,
    'login with other = connect with other'	:28,
    'type email'				:29,	
    'type phone number'				:30,
    'back button'				:31,
    'login with email'				:32,
    'login link'				:33,
    'sign in link'				:34,
    'contact link'				:35,
    'reset button'				:36,
    'login with yahoo'				:37,
    'login with instagram'			:38,
    'download link'				:39,
    'next page'					:40,
    'previous page'				:41,
    'enter'					:42,
    'contact link'				:43,
    'advertisement'				:44,
    'heading'					:45,
    'sub heading'				:46,
    'writings'					:47,
    'image'					:48,
    'selection'					:49,
    'video'					:50,
    'fill the box'				:51,
    'Address field'				:52,
    'DOB'					:53,
    'sex'					:54,
    'join/sign up'				:55,
    'tick box if necessary'			:56,
    'fill the box'				:57,
    'connect with whatsapp'			:58

}


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


class PascalVocGenerator(Generator):
    """ Generate data for a Pascal VOC dataset.

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
        self,
        data_dir,
        set_name,
        classes=voc_classes,
        image_extension='.png',
        skip_truncated=False,
        skip_difficult=False,
        **kwargs
    ):
        """ Initialize a Pascal VOC data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            csv_class_file: Path to the CSV classes file.
        """
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.classes              = classes
        self.image_names          = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(PascalVocGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        return read_image_bgr(path)

    def __parse_annotation(self, element):
        """ Parse an annotation given an XML element.
        """
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox    = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box, label

    def __parse_annotations(self, xml_root):
        """ Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((len(xml_root.findall('object')),)), 'bboxes': np.empty((len(xml_root.findall('object')), 4))}
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box, label = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue

            annotations['bboxes'][i, :] = box
            annotations['labels'][i] = label

        return annotations

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
