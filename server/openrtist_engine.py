# OpenRTiST
#   - Real-time Style Transfer
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#           Shilpa George <shilpag@andrew.cmu.edu>
#           Thomas Eiszler <teiszler@andrew.cmu.edu>
#           Padmanabhan Pillai <padmanabhan.s.pillai@intel.com>
#           Roger Iyengar <iyengar@cmu.edu>
#
#   Copyright (C) 2011-2019 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
# Portions of this code borrow from sample code distributed as part of
# Intel OpenVino, which is also distributed under the Apache License.
#
# Portions of this code were modified from sampled code distributed as part of
# the fast_neural_style example that is part of the pytorch repository and is
# distributed under the BSD 3-Clause License.
# https://github.com/pytorch/examples/blob/master/LICENSE

import cv2
import numpy as np
import logging
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
import openrtist_pb2

from emotion_to_style import emotion_to_style_map
from io import BytesIO
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person


logger = logging.getLogger(__name__)

face_supported = False

try:
    from azure_face_credentials import ENDPOINT, KEY
    face_supported = True
except ImportError:
    logger.info("AZURE FACE IS NOT SUPPORTED")


class OpenrtistEngine(cognitive_engine.Engine):
    SOURCE_NAME = "openrtist"


    def __init__(self, compression_params, adapter):
        self.compression_params = compression_params
        self.adapter = adapter

        # Emotion enabled
        if face_supported:
            self.face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
        
        

        # The waterMark is of dimension 30x120
        wtr_mrk4 = cv2.imread("../wtrMrk.png", -1)

        # The RGB channels are equivalent
        self.mrk, _, _, mrk_alpha = cv2.split(wtr_mrk4)

        self.alpha = mrk_alpha.astype(float) / 255

        # TODO support server display

        logger.info("FINISHED INITIALISATION")

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        extras = cognitive_engine.unpack_extras(openrtist_pb2.Extras, input_frame)

        # Preprocessing steps used by both engines
        np_data = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        orig_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)


        new_style = False
        send_style_list = False
        emotion_enabled = False

        if extras.style == "?":
            new_style = True
            send_style_list = True
        elif extras.style == 'emotion_enabled':
            logger.info("EMOTION-ENABLED")
            emotion_enabled = True
            style = self.emotion_detection(orig_img)
            if style:
                print('style' + style)
                self.adapter.set_style(style)
                new_style = True
        elif extras.style != self.adapter.get_style():
            # if user just changed the style, set to new style
            self.adapter.set_style(extras.style)
            logger.info("New Style: %s", extras.style)
            new_style = True

        if not emotion_enabled:
            style = self.adapter.get_style()

        # if no face detected and no style is selected, then bypass processing
        if style:
            image = self.process_image(orig_img)
        else:
            image = orig_img

        # depth
        image = image.astype("uint8")
        if extras.HasField("depth_map"):
            # protobuf contains depth_map
            depth_map = extras.depth_map.value
            # get depth map (bytes) and perform depth thresholding to create foreground mask with 3 channels
            depth_threshold = extras.depth_threshold

            # data type conversion from bytes to a scaled-out 2d numpy array (480*640)
            np_depth_1d = np.frombuffer(depth_map, dtype=np.uint16)
            np_depth_2d = np.reshape(np_depth_1d, (-1, 160))

            # threshold on the distance
            mask_fg = cv2.inRange(np_depth_2d, 0, depth_threshold)

            # resize to match the image
            orig_h, orig_w, _ = orig_img.shape
            mask_fg = cv2.resize(
                mask_fg, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

            # Apply morphology to the thresholded image to remove extraneous white regions and save a mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel)
            mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel)

            fg = cv2.bitwise_and(orig_img, orig_img, mask=mask_fg)

            # get background mask by inversion
            mask_bg = cv2.bitwise_not(mask_fg)

            # get background from transformed image
            bg = cv2.bitwise_and(image, image, mask=mask_bg)

            # stitch transformed background and original foreground
            image = cv2.bitwise_or(fg, bg)

        image = self._apply_watermark(image)

        _, jpeg_img = cv2.imencode(".jpg", image, self.compression_params)
        img_data = jpeg_img.tostring()

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data

        extras = openrtist_pb2.Extras()
        if style:
            extras.style = style

        if new_style:
            extras.style_image.value = self.adapter.get_style_image()
        if send_style_list:
            if face_supported:
                extras.style_list['emotion_enabled'] = 'switch styles based on your emotion'
            for k, v in self.adapter.get_all_styles().items():
                extras.style_list[k] = v

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.results.append(result)
        result_wrapper.extras.Pack(extras)

        return result_wrapper


    def emotion_detection(self, orig_img):
        # detect_with_stream:
        # Up to 100 faces can be returned for an image. Faces are ranked by face rectangle size from large to small.
        # JPEG, PNG, GIF (the first frame), and BMP format are supported. The allowed image file size is from 1KB to 6MB.
        is_success, im_buf_arr = cv2.imencode(".bmp", orig_img)
        image_jpg = BytesIO(im_buf_arr)
        face_attributes = ['emotion']
        detected_faces = self.face_client.face.detect_with_stream(image_jpg, return_face_attributes=face_attributes)

        if not detected_faces:
            print('DEBUGHERE ERROR: No face detected')
            return None
        else:
            def get_emotion(emoObject):
                emoDict = dict()
                emoDict['anger'] = emoObject.anger
                emoDict['contempt'] = emoObject.contempt
                emoDict['disgust'] = emoObject.disgust
                emoDict['fear'] = emoObject.fear
                emoDict['happiness'] = emoObject.happiness
                emoDict['neutral'] = emoObject.neutral
                emoDict['sadness'] = emoObject.sadness
                emoDict['surprise'] = emoObject.surprise
                emo_name = max(emoDict, key=emoDict.get)
                emo_level = emoDict[emo_name]
                return emo_name, emo_level

            # get the largest face in the image
            largest_face = detected_faces[0]
            emotion, confidence = get_emotion(largest_face.face_attributes.emotion)
            print("DEBUGHERE: {} emotion with confidence level {}".format(emotion, confidence))

            if emotion in emotion_to_style_map:
                return emotion_to_style_map[emotion]
            else:
                return None

    def process_image(self, image):
        preprocessed = self.adapter.preprocessing(image)
        post_inference = self.inference(preprocessed)
        img_out = self.adapter.postprocessing(post_inference)
        return img_out

    def inference(self, preprocessed):
        """Allow timing engine to override this"""
        return self.adapter.inference(preprocessed)

    def _apply_watermark(self, image):
        img_mrk = image[-30:, -120:]  # The waterMark is of dimension 30x120
        img_mrk[:, :, 0] = (1 - self.alpha) * img_mrk[:, :, 0] + self.alpha * self.mrk
        img_mrk[:, :, 1] = (1 - self.alpha) * img_mrk[:, :, 1] + self.alpha * self.mrk
        img_mrk[:, :, 2] = (1 - self.alpha) * img_mrk[:, :, 2] + self.alpha * self.mrk
        image[-30:, -120:] = img_mrk
        # img_out = image.astype("uint8")
        img_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return img_out
