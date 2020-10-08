import os
from typing import Optional, Tuple, List, Type
import numpy as np
from cv2 import cvtColor, COLOR_BGR2GRAY, imshow, waitKey, imread, ORB_create, ORB, KeyPoint, drawMarker, MARKER_CROSS, \
    COLOR_GRAY2BGR, BFMatcher, NORM_HAMMING, DMatch, drawMatches, \
    Rodrigues, resize, INTER_AREA, solvePnPRansac, projectPoints, \
    goodFeaturesToTrack, calcOpticalFlowPyrLK, TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT
from recordclass import RecordClass
from collections import namedtuple
from abc import ABC, abstractmethod
from enum import Enum
import glob

radiansToDegrees: float = 1 / np.pi * 180.0
EulerAngles = namedtuple('EulerAngles', ['yaw_rad', 'pitch_rad', 'roll_rad'])


def get_intrinsic_matrix(focal_length_x_pixel: float,
                         focal_length_y_pixel: float,
                         skew: float,
                         principal_point_x_pixel: float,
                         principal_point_y_pixel: float) -> np.ndarray:
    """
    This function constructs an intrinsic calibration matrix, given explicit parameters
    """
    return np.array([[focal_length_x_pixel, skew, principal_point_x_pixel],
                     [.0, focal_length_y_pixel, principal_point_y_pixel],
                     [.0, .0, 1.0]])


def get_euler_angles_from_rotation_matrix_ZYX_order(rotation_matrix: np.ndarray) -> EulerAngles:
    """
    Retrieve euler angles from rotation matrix (direction cosine matrix)
    assuming Z-Y-X order of rotation
    reference: D.H.Titterton, Strapdown Inertial Navigation, (3.66)
    @param rotation_matrix, 3X3
    @return: out[0] - heading euler angle [rad]
             out[1] - pitch euler angle [rad]
             out[2] - roll euler angle [rad]
    """
    roll_rad = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])
    pitch_rad = np.arcsin(-rotation_matrix[2][0])
    yaw_rad = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])
    return EulerAngles(yaw_rad=yaw_rad,
                       pitch_rad=pitch_rad,
                       roll_rad=roll_rad)


class ProcessedFrameData(RecordClass):
    """
    This object aggregates frame + extracted features (keypoints and descriptors) into a single data object
    """
    frame: np.ndarray
    list_of_keypoints: List[KeyPoint]
    descriptors: np.ndarray

    @classmethod
    def build(cls,
              frame: np.ndarray,
              list_of_keypoints: List[KeyPoint],
              descriptors: np.ndarray):
        return ProcessedFrameData(frame,
                                  list_of_keypoints,
                                  descriptors)


class RobotPose(RecordClass):
    """
    This objects contains the robot pose (horizontal translation + yaw angle)
    """
    position_x_meter: float
    position_y_meter: float
    yaw_angle_deg: float

    @classmethod
    def build(cls,
              position_x_meter: float,
              position_y_meter: float,
              yaw_angle_deg: float):
        return RobotPose(position_x_meter,
                         position_y_meter,
                         yaw_angle_deg)


class FeaturesHandlerAbstractBase(ABC):
    """
    This is an abstract base class for a features extractor
    """

    features_extractor: object
    features_matcher: object

    @abstractmethod
    def extract_features(self, frame: np.ndarray) -> ProcessedFrameData:
        """
        This method extracts features from a frame
        Returns
        -------
        """
        pass

    @abstractmethod
    def match_features(self, frame_1: ProcessedFrameData, frame_2: ProcessedFrameData):
        """
        This method matches features between frames
        """

    @abstractmethod
    def is_handler_capable(self, frame: np.ndarray) -> bool:
        """
        This method tries to measure this handlers capability to track features for an unknown scene type
        (i.e illumination, texture, etc)
        """


class OrbFeaturesHandler(FeaturesHandlerAbstractBase):
    """
    This class implements an ORB features exractor
    """
    features_extractor: ORB
    features_matcher: BFMatcher

    DEFAULT_DISTANCE_THRESHOLD_FOR_SUCCESSFULL_FEATURE_MATCH: int = 10

    DEFAULT_nfeatures = 1000

    def __init__(self):
        self.features_extractor = ORB_create(nfeatures=self.DEFAULT_nfeatures)

        # ORB uses binary descriptors -> use hamming norm (XOR between descriptors)
        self.features_matcher = BFMatcher(NORM_HAMMING, crossCheck=True)

    def extract_features(self, frame: np.ndarray) -> ProcessedFrameData:
        """
        This method extracts ORB features
        """
        list_of_keypoints, descriptors = self.features_extractor.detectAndCompute(image=frame,
                                                                                  mask=None)
        return ProcessedFrameData.build(frame=frame,
                                        list_of_keypoints=list_of_keypoints,
                                        descriptors=descriptors)

    def match_features(self, frame_1: ProcessedFrameData, frame_2: ProcessedFrameData):
        """
        This method matches ORB features
        based on https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        """
        list_of_matches: List[DMatch] = self.features_matcher.match(queryDescriptors=frame_1.descriptors,
                                                                    trainDescriptors=frame_2.descriptors)
        return sorted(list_of_matches, key=lambda x: x.distance)

        # # Sort them in the order of their distance.
        # return

    def is_handler_capable(self, frame: np.ndarray) -> bool:
        """
        This method implements ORB handler capability test
        """
        extracted_features: ProcessedFrameData = self.extract_features(frame=frame)
        return len(extracted_features.list_of_keypoints) >= int(0.9 * self.DEFAULT_nfeatures)


class ShiTomasiFeaturesHandler(FeaturesHandlerAbstractBase):
    """
    This class implements an Shi-Tomasi edge-detector features exractor
    """
    features_extractor: object = None
    features_matcher: object = None

    DEFAULT_DISTANCE_THRESHOLD_FOR_SUCCESSFULL_FEATURE_MATCH: int = 10

    DEFAULT_nfeatures = 1000

    def extract_features(self, frame: np.ndarray) -> ProcessedFrameData:
        """
        This method extracts ORB features
        """
        res = goodFeaturesToTrack(image=frame,
                                  maxCorners=self.DEFAULT_nfeatures,
                                  qualityLevel=0.01,
                                  minDistance=10)
        list_of_keypoints: List[KeyPoint] = [KeyPoint(point[0], point[1], -1) for point in list(res.squeeze())]
        return ProcessedFrameData.build(frame=frame,
                                        list_of_keypoints=list_of_keypoints,
                                        descriptors=res)

    def is_handler_capable(self, frame: np.ndarray) -> bool:
        """
        This method implements Shi-Tomasi handler capability test
        """
        extracted_features: ProcessedFrameData = self.extract_features(frame=frame)
        return len(extracted_features.list_of_keypoints) >= int(0.9 * self.DEFAULT_nfeatures)

    def match_features(self, frame_1: ProcessedFrameData, frame_2: ProcessedFrameData):
        """
        This method uses sparse optical-flow (KLT algorithm)
        https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
        """
        lk_params = dict(winSize=(5, 5),
                         maxLevel=2,
                         criteria=(TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))
        prev_points: np.ndarray = np.array([(point.pt[0], point.pt[1]) for point in frame_1.list_of_keypoints]).astype(
            dtype=np.float32)
        new_points, st, err = calcOpticalFlowPyrLK(frame_1.frame,
                                                   frame_2.frame,
                                                   prev_points,
                                                   None,
                                                   **lk_params)

        list_of_matches: List[DMatch] = list()
        for idx, data in enumerate(zip(st.squeeze(), err.squeeze())):
            status = data[0]
            err = data[1]
            if status == 1:
                queryIdx = idx
                trainIdx = idx
                distance = err
                list_of_matches.append(DMatch(queryIdx, trainIdx, distance))
        return list_of_matches


class MotionEstimationStatus(Enum):
    """
    This enum indicates the motion estimation algorithm status
    """
    MOTION_WAITING_FOR_FIRST_FRAME = 0
    MOTION_NOT_UPDATED = 1
    MOTION_OK = 2
    MOTION_ALGORITHM_INCAPABLE = 3


class RobotHorizontalMotionEstimator:
    """
    This class tracks the horizontal motion of a robot, navigating above a ground-plane,
    using an incoming video stream from a down-facing camera
    """
    robot_height_in_meter: float
    camera_intrinsic: np.ndarray

    features_handler: Type[FeaturesHandlerAbstractBase]
    previous_frame_data: Optional[ProcessedFrameData]
    current_frame_number: int

    current_robot_pose: Optional[RobotPose]
    motion_estimation_status: MotionEstimationStatus

    debug_flag: bool
    MARKER_TYPE_FOR_DEBUG: int = MARKER_CROSS
    MARKER_SIZE_FOR_DEBUG: int = 5

    RESIZE_RATIO_FOR_DEBUG: int = 2

    REPROJECTION_TOLERANCE: float = 1e-10

    def __init__(self, robot_height_in_meter: float,
                 focal_length_x_pixel: float,
                 focal_length_y_pixel: float,
                 skew: float,
                 principal_point_x_pixel: float,
                 principal_point_y_pixel: float,
                 debug_flag: bool = False):
        self.robot_height_in_meter = robot_height_in_meter
        self.camera_intrinsic = get_intrinsic_matrix(focal_length_x_pixel=focal_length_x_pixel,
                                                     focal_length_y_pixel=focal_length_y_pixel,
                                                     skew=skew,
                                                     principal_point_x_pixel=principal_point_x_pixel,
                                                     principal_point_y_pixel=principal_point_y_pixel)
        self.camera_intrinsic_inv = np.linalg.inv(self.camera_intrinsic)
        self.current_frame_number = 0
        self.init_motion_estimation()
        self.debug_flag = debug_flag

    def init_motion_estimation(self):
        """
        This method initializes the motion estimation block
        """
        self.previous_frame_data = None
        self.motion_estimation_status = MotionEstimationStatus.MOTION_WAITING_FOR_FIRST_FRAME
        self.current_robot_pose = None

    def init_features_handler(self, prepared_frame: np.ndarray):
        """
        This method attempts to choose a features extractor, for an unknown environment
        """
        # try ORB features handler
        self.features_handler = OrbFeaturesHandler()
        if self.features_handler.is_handler_capable(frame=prepared_frame):
            # ORB is capable of handling current scene
            print("Selecting ORB features based motion tracker")
            return

        # try Shi-Tomasi
        self.features_handler = ShiTomasiFeaturesHandler()
        if self.features_handler.is_handler_capable(frame=prepared_frame):
            # Shi-Tomasi is capable of handling current scene
            print("Selecting Shi-Tomasi features based motion tracker")
            return

        # could not find suitable features handler
        self.motion_estimation_status = MotionEstimationStatus.MOTION_ALGORITHM_INCAPABLE

    def set_new_frame(self, raw_frame: np.ndarray):
        """
        This method passes a new frame to the motion estimation class
        Parameters
        ----------
        raw_frame

        Returns
        -------

        """
        self.current_frame_number += 1
        prepared_frame: np.ndarray = self.prepare_frame(frame=raw_frame)
        if self.motion_estimation_status in (MotionEstimationStatus.MOTION_WAITING_FOR_FIRST_FRAME,
                                             MotionEstimationStatus.MOTION_ALGORITHM_INCAPABLE):
            self.init_features_handler(prepared_frame)
        try:
            self.update_motion_using_new_frame(frame=prepared_frame)
        except:
            self.motion_estimation_status = MotionEstimationStatus.MOTION_NOT_UPDATED
            return

    def prepare_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        This method prepares the received frame for subsequent processing
        Returns
        -------

        """
        return cvtColor(frame, COLOR_BGR2GRAY)

    def process_frame(self, frame: np.ndarray) -> ProcessedFrameData:
        """
        This method extracts features from a frame and returns an aggregated ProcessedFrameData object
        """
        return self.features_handler.extract_features(frame=frame)

    def update_motion_using_new_frame(self, frame: np.ndarray):
        """
        This method
        Parameters
        ----------
        frame

        Returns
        -------

        """
        processed_frame: ProcessedFrameData = self.process_frame(frame=frame)
        if self.debug_flag:
            # draw features keypoints
            temp_frame: np.ndarray = cvtColor(processed_frame.frame, COLOR_GRAY2BGR)
            for keypoint in processed_frame.list_of_keypoints:
                drawMarker(img=temp_frame,
                           position=(int(keypoint.pt[0]), int(keypoint.pt[1])),
                           color=(0, 255, 0),
                           markerSize=self.MARKER_SIZE_FOR_DEBUG,
                           markerType=self.MARKER_TYPE_FOR_DEBUG)
            imshow(winname="frame #{0:d} with detected features".format(self.current_frame_number),
                   mat=self.resize_image(temp_frame))

        if self.previous_frame_data is None:
            # first frame, cannot estimate motion
            print("Robot motion estimation initialized")
            self.motion_estimation_status = MotionEstimationStatus.MOTION_NOT_UPDATED
        else:
            self.update_motion(new_processed_frame_data=processed_frame)
        self.previous_frame_data = processed_frame

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        This method resizes an image, for visualization
        """
        scale_percent = int(100 / self.RESIZE_RATIO_FOR_DEBUG)  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        return resize(image, dim, interpolation=INTER_AREA)

    def update_motion(self, new_processed_frame_data: ProcessedFrameData):
        """
        This method estimates the horizontal translation of a robot, navigating above a ground-plance,
        given two down-facing frames captured by the robot's camera.
        The robot's altitude above the ground-plane is assumed to be known and constant
        """
        # match features
        list_of_matches: List[DMatch] = self.features_handler.match_features(frame_1=self.previous_frame_data,
                                                                             frame_2=new_processed_frame_data)
        list_of_matches = list_of_matches[:100]
        if self.debug_flag:
            imshow("feature matches frame #{0:d}->#{1:d}".format(self.current_frame_number-1,
                                                                 self.current_frame_number), mat=self.resize_image(drawMatches(self.previous_frame_data.frame,
                                                                        self.previous_frame_data.list_of_keypoints,
                                                                        new_processed_frame_data.frame,
                                                                        new_processed_frame_data.list_of_keypoints,
                                                                        list_of_matches,
                                                                        None,
                                                                        flags=2)))
        self.get_camera_motion_using_PnP(list_of_matches,
                                         new_processed_frame_data=new_processed_frame_data)

    def print_current_position(self):
        """
        This method prints the current position of the robot, if available
        """
        if self.motion_estimation_status == MotionEstimationStatus.MOTION_OK:
            print("frame #{0:d} | Robot motion: X = {1:f} [m] Y = {2:f} [m] | rotation: yaw = {3:f} [deg]".format(
                self.current_frame_number,
                self.current_robot_pose.position_x_meter,
                self.current_robot_pose.position_y_meter,
                self.current_robot_pose.yaw_angle_deg))
        elif self.motion_estimation_status == MotionEstimationStatus.MOTION_WAITING_FOR_FIRST_FRAME:
            print("frame #{0:d} | Robot motion unavailable - waiting for 1-st frame".format(self.current_frame_number))
        elif self.motion_estimation_status == MotionEstimationStatus.MOTION_NOT_UPDATED:
            print("frame #{0:d} | Robot motion could not be updated".format(self.current_frame_number))
        elif self.motion_estimation_status == MotionEstimationStatus.MOTION_ALGORITHM_INCAPABLE:
            print("frame #{0:d} | Robot motion unavailable - cannot handle this type of scene".format(
                self.current_frame_number))

    def get_camera_motion_using_PnP(self, list_of_matches: List[DMatch],
                                    new_processed_frame_data: ProcessedFrameData):
        """
        This method estimates camera motion using the following steps:
        1. receive a set of 2D features, matched between the 1-st and the 2-nd frame
        2. (3d mapping) localize 3D location of features, using their 2d projection and camera height
        3. (camera localization) solve the 2-nd frame camera pose, with the help of the PnP algorithm, by using the
        mapped 3D locations of the features and their 2-nd frame 2d projections

        Estimation of 3D feature location, w.r.t to the 1-st camera frame, is performed by inverting their projection in the
        1-st camera frame + known camera height (H):
        p_cam_homogeneous = K * [R | t] * P_world_homogeneous
        where:
        p_cam_homogeneous - homogeneous coordinates of feature in frame
        P_world_homogeneous - homogeneous coordinates of feature in 3D world

        p_cam_homogeneous = s * [u, v, 1]'
        s * [u, v, 1]' = K * [I | 0] * [X Y H 1]' (R=I and t=0, for 1-st camera w.r.t 1-st camera frame)
                      = K * [X Y H]'
        -> s = H (scale/depth of scene, extracted from 3-rd component)
        H * [u, v, 1]' = K * [X Y H]'
        -> [X Y H]' = H * K^(-1) * [u, v, 1]'

        vectorize for all features:
        [X_1 X_2 ... X_N                   [u_1 u_2 ... u_N
         Y_1 Y_2 ... Y_N   =  H * K^(-1) *  v_1 v_2 ... v_N
         H   H   ... H  ]                     1     ...   1]
        """

        N: int = len(list_of_matches)
        H: float = self.robot_height_in_meter
        K_inv: np.ndarray = self.camera_intrinsic_inv

        # construct array of feature locations, in previous frame (Nx2)
        array_of_2d_coords_previous_frame = np.array(
            [self.previous_frame_data.list_of_keypoints[match.queryIdx].pt for match in list_of_matches])

        # estimate array of 3D feature locations, w.r.t previous camera frame (3xN)
        estimated_array_of_3d_coords_previous_frame = H * K_inv @ np.array(
            np.vstack((array_of_2d_coords_previous_frame.T,
                       np.ones((1, N)))))
        # transpose to Nx3, for future use by opencv methods
        estimated_array_of_3d_coords_previous_frame = estimated_array_of_3d_coords_previous_frame.T

        # construct array of matched feature locations, in new frame (Nx2)
        array_of_2d_coords_new_frame = np.array(
            [new_processed_frame_data.list_of_keypoints[match.trainIdx].pt for match in list_of_matches])

        # check re-projection of 3D points to 2D pixels
        re_project, _ = projectPoints(objectPoints=estimated_array_of_3d_coords_previous_frame,
                                      rvec=np.zeros(3),
                                      tvec=np.zeros(3),
                                      cameraMatrix=self.camera_intrinsic,
                                      distCoeffs=None)

        max_reprojection_error = np.max(np.abs(re_project.squeeze() - array_of_2d_coords_previous_frame))
        if max_reprojection_error > self.REPROJECTION_TOLERANCE:
            print("3D re-projection fail, aborting motion estimation")
            self.motion_estimation_status = MotionEstimationStatus.MOTION_NOT_UPDATED
            return

        return_flag, est_rotation_vector, est_translation, inliers = solvePnPRansac(
            objectPoints=estimated_array_of_3d_coords_previous_frame,
            imagePoints=array_of_2d_coords_new_frame,
            cameraMatrix=self.camera_intrinsic,
            distCoeffs=None)

        if not return_flag:
            print("PnP fail, aborting motion estimation")
            self.motion_estimation_status = MotionEstimationStatus.MOTION_NOT_UPDATED

        # check amount of outliers from RANSAC
        if len(inliers) < int(N * 0.9):
            # TODO useful number
            print("PnP RANSAC fail, aborting motion estimation")
            self.motion_estimation_status = MotionEstimationStatus.MOTION_NOT_UPDATED
            return

        # convert rotation vector to rotation matrix, to extract yaw angle
        euler_angles: EulerAngles = get_euler_angles_from_rotation_matrix_ZYX_order(
            Rodrigues(src=est_rotation_vector)[0])

        self.motion_estimation_status = MotionEstimationStatus.MOTION_OK

        # we have recovered the pose of the camera in the 2-nd frame
        # TODO fix motion definition
        self.current_robot_pose = RobotPose.build(position_x_meter=(-1) * est_translation[0][0],
                                                  position_y_meter=(-1) * est_translation[1][0],
                                                  yaw_angle_deg=(-1)* radiansToDegrees * euler_angles.yaw_rad)


if __name__ == "__main__":
    current_dir: str = os.getcwd()
    list_of_frame_paths: List[str] = list()
    test_dir_str: str = os.path.join(current_dir,'test_mosaic_height')

    robot_height_in_meter = 2.5
    robot_motion_estimator: RobotHorizontalMotionEstimator = RobotHorizontalMotionEstimator(
        robot_height_in_meter=robot_height_in_meter,
        focal_length_x_pixel=1422.0,
        focal_length_y_pixel=1422.0,
        skew=.0,
        principal_point_x_pixel=1024.0 / 2,
        principal_point_y_pixel=768.0 / 2,
        debug_flag=False)

    for frame_path in sorted(glob.glob(pathname=os.path.join(test_dir_str,'*.png'))):
        frame: Optional[np.ndarray] = imread(frame_path)
        if frame is not None:
            robot_motion_estimator.set_new_frame(raw_frame=frame)
            robot_motion_estimator.print_current_position()
        else:
            print("Error! cannot read frame at location = {0:s}".format(frame_path))
            print("Aborting motion estimation")
            exit(1)
    waitKey(0)
    exit(0)
