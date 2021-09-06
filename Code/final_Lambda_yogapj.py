#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************

from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import mo
import greengrasssdk

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream"""
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

def infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        # is quite large so we upload them to a list to map the machine labels to human readable
        # labels.
        model_type = 'classification'
        with open('caltech256_labels.txt', 'r') as labels_file:
            output_map = [class_label.rstrip() for class_label in labels_file]
        # Create an IoT client for sending to messages to the cloud.
        client = greengrasssdk.client('iot-data')
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        # The height and width of the training set images
        input_height = 224
        input_width = 224
        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        model_name = "image-classification"
        ret, model_path = mo.optimize(model_name, input_width, input_height, aux_inputs={'--epoch': 5,'--precision':'FP16'})
        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading  model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload=' loaded')
        # The number of top results to stream to IoT.
        num_top_k = 3
        # Define the detection region size.
        region_size = 1500
        # Define the inference display region size. This size was decided based on the longest label.
        label_region_width = 590
        label_region_height = 400
        # Heading for the inference display.
        
        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Crop the detection region for inference.
            frame_crop = frame[int(frame.shape[0]/2-region_size/2):int(frame.shape[0]/2+region_size/2), \
                         int(frame.shape[1]/2-region_size/2):int(frame.shape[1]/2+region_size/2), :]
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame_crop, (input_height, input_width))
            # Model was trained in RGB format but getLastFrame returns image
            # in BGR format so need to switch.
            frame_resize = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a classification model,
            # a simple API is provided.
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            # Get top k results with highest probabilities
            top_k = parsed_inference_results[model_type][0:num_top_k]
            # Create a copy of the original frame.
            overlay = frame.copy()
            # Create the rectangle that shows the inference results.
            cv2.rectangle(overlay, (0, 0), \
                          (int(label_region_width), int(label_region_height)), (211,211,211), -1)
            # Blend with the original frame.
            opacity = 0.7
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
            # Add the header for the inference results.
            
            # Add the label along with the probability of the top result to the frame used by local display.
            # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
            # for more information about the cv2.putText method.
            # Method signature: image, text, origin, font face, font scale, color, and tickness
            if top_k[0]['prob']*100 < 60 :
                cv2.putText(frame,'Take your pose',(0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            else :
                cv2.putText(frame, output_map[top_k[0]['label']] + ' ' + str(round(top_k[0]['prob'], 3) * 100) + '%', \
                            (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
           
            # Display the detection region.
            cv2.rectangle(frame, (int(frame.shape[1]/2-region_size/2), int(frame.shape[0]/2-region_size/2)), \
                          (int(frame.shape[1]/2+region_size/2), int(frame.shape[0]/2+region_size/2)), (255,0,0), 5)
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send the top k results to the IoT console via MQTT
            cloud_output = {}
            for obj in top_k:
                cloud_output[output_map[obj['label']]] = obj['prob']
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error : {}'.format(ex))

infinite_infer_run()
