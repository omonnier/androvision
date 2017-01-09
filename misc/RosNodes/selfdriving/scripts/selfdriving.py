from __future__ import print_function

import pandas as pd
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import constants
from sensor_listener import SensorListener


class TrainModel(SensorListener):
    def __init__(self, dataset_file):
        SensorListener.__init__(self)
        # TODO: include /linear.x value in the dataset.
        #self._sensor_values.update({LINEAR_KEY: 0.0, ANGULAR_KEY: 0.0})
        self._sensor_values.update({constants.ANGULAR_KEY: 0.0})
        self._dataset_file = dataset_file
        self._data_frame = pd.DataFrame(columns=sorted(self._sensor_values.keys()))

    def init(self):
        SensorListener.init(self)
        # Init listener to store commands from user during training phase
        rospy.Subscriber(constants.VELOCITY_ACTION_TOPIC, Twist, self.command_callback)

    def command_callback(self, data):
        # TODO: include /linear.x value in the dataset.
        #self._sensor_values[LINEAR_KEY] = data.linear.x
        self._sensor_values[constants.ANGULAR_KEY] = data.angular.z
        self._data_frame.loc[len(self._data_frame)] = [self._sensor_values[x] for x in sorted(self._sensor_values.keys())]
        self._data_frame.to_csv(self._dataset_file)

class PredictModel(SensorListener):
    def __init__(self, dataset_file, model_to_use=None):
        SensorListener.__init__(self)
        if model_to_use is None:
            model_to_use = DecisionTreeRegressor()
        self._model = model_to_use
        self._scaler = None
        self._dataset_file = dataset_file
        self._command_publisher = None
        self._feature_columns = None
        # Build the model
        self.load_model()

    def load_model(self):
        self._scaler = joblib.load('scaler.pkl')
        self._model = joblib.load('model.pkl')
        # hack: need to hardcode the feature used to train the model
        self._feature_columns = ["/left", "/right"]

    def init(self):
        SensorListener.init(self)
        # Init publisher to send predicted command
        self._command_publisher = rospy.Publisher(constants.VELOCITY_ACTION_TOPIC, Twist, queue_size=10)

    def X_scaling(self, scaler, values):
        # add dummy y value
        values.add(0.)
        scaled_values = scaler.transform(values)
        # remove y scaled value
        return scaled_values[:-1]

    def Y_inverse_scaling(self, scaler, value):
        # add dummy X values for left, right, delta
        values = [0., 0., 0.]  + value
        return scaler.inverse_transform(values)[-1]

    def sonar_callback(self, data):
        SensorListener.sonar_callback(self, data)
        if self._command_publisher:
            feature_values = [self._sensor_values.get(x) for x in self._feature_columns if x in self._sensor_values]
            # hack: the model was trained with the delta: left - right
            feature_values.append(self._sensor_values.get("/left") - self._sensor_values.get("/right"))
            feature_values = self.X_scaling(feature_values)
            predict_val = self._model.predict([feature_values])[0]
            predict_val = self.Y_inverse_scaling(predict_val)
            linear_x = 1 if predict_val != 0 else 0
            cmd_vel = Twist(Vector3(linear_x, 0, 0), Vector3(0, 0, predict_val))
            self._command_publisher.publish(cmd_vel)

if __name__ == '__main__':
    model = RandomForestRegressor(n_estimators=250)
    rosnode = PredictModel(dataset_file=constants.DATASET_FILE, model_to_use=model)
    try:
        rosnode.init()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
