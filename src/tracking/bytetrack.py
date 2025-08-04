import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from loguru import logger
import cv2


class TrackState:
    """Track state enumeration."""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for tracking objects."""
    
    _count = 0
    
    track_id = 0
    is_activated = False
    state = TrackState.New
    
    history = defaultdict(list)
    
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count
        
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        self.is_activated = True
        self.state = TrackState.Tracked
        
        if frame_id == 1:
            self.is_activated = True
            
    def predict(self):
        """Predict next state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # Set velocity to 0 for inactive tracks
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    def update(self, new_track, frame_id):
        """Update track with new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        
        self.score = new_track.score
        self.class_name = new_track.class_name
        
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
        
    def to_xyah(self):
        """Get current position in xyah format."""
        return self.mean[:4].copy()
        
    def to_tlwh(self):
        """Get current position in tlwh format."""
        ret = self.to_xyah().copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
        
    def to_tlbr(self):
        """Get current position in tlbr format."""
        ret = self.to_tlwh().copy()
        ret[2:] = ret[:2] + ret[2:]
        return ret
        
    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class STrack(BaseTrack):
    """Single object track for ByteTrack."""
    
    def __init__(self, tlwh, score, class_name="person", temp_feat=None, buffer_size=30):
        """
        Initialize STrack.
        
        Args:
            tlwh: Bounding box in tlwh format [top, left, width, height]
            score: Detection confidence score
            class_name: Object class name
            temp_feat: Temporary feature (unused in this implementation)
            buffer_size: History buffer size
        """
        # Wait activation
        self.xywh = None  # (cx, cy, w, h)
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = score
        self.class_name = class_name
        
        self.tracklet_len = 0
        self.smooth_feat = None
        self.features = []
        self.alpha = 0.9
        self.mean = None
        self.covariance = None
        
        self.is_activated = False
        self.track_id = 0
        self.state = TrackState.New
        
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0
        
        self.time_since_update = 0
        self.location = (np.inf, np.inf)
        
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self.tlwh))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        
        if frame_id == 1:
            self.is_activated = True
            
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        
        if new_id:
            self.track_id = self.next_id()
            
        self.score = new_track.score
        self.class_name = new_track.class_name
        
    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.Lost
        
    def mark_removed(self):
        """Mark track as removed."""
        self.state = TrackState.Removed


class KalmanFilter:
    """Kalman filter for object tracking."""
    
    def __init__(self):
        ndim, dt = 4, 1.0
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
            
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        
    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
        
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
            
        return mean, covariance
        
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
            
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        projected_cov += innovation_cov
        
        # Simplified Kalman gain calculation
        try:
            kalman_gain = np.linalg.solve(
                projected_cov, 
                np.dot(covariance, self._update_mat.T).T
            ).T
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            kalman_gain = np.dot(
                np.dot(covariance, self._update_mat.T),
                np.linalg.pinv(projected_cov)
            )
            
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
            
        return new_mean, new_covariance


class ByteTracker:
    """ByteTrack multi-object tracker."""
    
    def __init__(self, 
                 track_thresh: float = 0.6,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Detection threshold for tracking
            track_buffer: Frames to keep lost tracks
            match_thresh: Matching threshold for associations
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.kalman_filter = KalmanFilter()
        
        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks
        self.removed_stracks = []  # Removed tracks
        
        self.frame_id = 0
        self.max_time_lost = int(self.frame_rate / 30.0 * self.track_buffer)
        
        logger.info(f"ByteTracker initialized with track_thresh={track_thresh}")
        
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame
            
        Returns:
            List of tracking results with track IDs
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Convert detections to STrack objects
        if len(detections) > 0:
            # Filter detections by confidence
            remain_inds = [i for i, det in enumerate(detections) 
                          if det['confidence'] >= self.track_thresh]
            
            dets = [detections[i] for i in remain_inds]
            detections_high = [self._detection_to_strack(det) for det in dets]
        else:
            detections_high = []
            
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        # Step 2: First association, with high confidence detections
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict the current location with KF
        for strack in strack_pool:
            strack.predict()
            
        dists = matching.iou_distance(strack_pool, detections_high)
        dists = matching.fuse_score(dists, detections_high)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(detections_high[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        # Second association, with low confidence detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections_second = []
        r_tracked_stracks_second = []
        
        matches, u_track, u_detection_second = matching.linear_assignment(
            matching.iou_distance(r_tracked_stracks_second, detections_second), thresh=0.5)
            
        for itracked, idet in matches:
            track = r_tracked_stracks_second[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        for it in u_track:
            track = r_tracked_stracks_second[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections_remain = [detections_high[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections_remain)
        matches, u_unconfirmed, u_detection_remain = matching.linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        # Step 4: Init new stracks
        for inew in u_detection_remain:
            track = detections_remain[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            
        # Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
            
        # Convert tracks to output format
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        tracking_results = []
        for track in output_stracks:
            tlbr = track.to_tlbr()
            
            tracking_result = {
                "frame_id": self.frame_id,
                "track_id": track.track_id,
                "bbox": tlbr.tolist(),  # [x1, y1, x2, y2]
                "class_name": track.class_name,
                "confidence": track.score
            }
            tracking_results.append(tracking_result)
            
        return tracking_results
        
    def _detection_to_strack(self, detection: Dict) -> STrack:
        """Convert detection dict to STrack object."""
        bbox = detection["bbox"]  # [x1, y1, x2, y2]
        
        # Convert to tlwh format
        tlwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        
        return STrack(
            tlwh=tlwh,
            score=detection["confidence"],
            class_name=detection["class_name"]
        )


# Helper functions for track management
def joint_stracks(tlista, tlistb):
    """Join two track lists."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Subtract track list b from track list a."""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks."""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


# Simplified matching module
class matching:
    """Simplified matching utilities."""
    
    @staticmethod
    def iou_distance(atracks, btracks):
        """Compute IoU distance matrix."""
        if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
           (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.to_tlbr() for track in atracks]
            btlbrs = [track.to_tlbr() for track in btracks]
        _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
        if _ious.size == 0:
            return _ious

        for i, atlbr in enumerate(atlbrs):
            for j, btlbr in enumerate(btlbrs):
                _ious[i, j] = bbox_iou(atlbr, btlbr)
        cost_matrix = 1 - _ious
        return cost_matrix
        
    @staticmethod
    def fuse_score(cost_matrix, detections):
        """Fuse detection scores with cost matrix."""
        if cost_matrix.size == 0:
            return cost_matrix
        iou_sim = 1 - cost_matrix
        det_scores = np.array([det.score for det in detections])
        det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
        fuse_sim = iou_sim * det_scores
        return 1 - fuse_sim
        
    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        """Perform linear assignment with Hungarian algorithm."""
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            matches = [[y[i], i] for i in x if i >= 0]
            unmatched_a = [i for i in range(cost_matrix.shape[0]) if x[i] < 0]
            unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in y]
        except ImportError:
            # Fallback to scipy if lap is not available
            from scipy.optimize import linear_sum_assignment
            matched_indices = linear_sum_assignment(cost_matrix)
            matches = []
            for i, j in zip(*matched_indices):
                if cost_matrix[i, j] <= thresh:
                    matches.append([i, j])
            unmatched_a = [i for i in range(cost_matrix.shape[0]) 
                          if i not in [m[0] for m in matches]]
            unmatched_b = [i for i in range(cost_matrix.shape[1]) 
                          if i not in [m[1] for m in matches]]
        return matches, unmatched_a, unmatched_b


def bbox_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # Calculate intersection
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
        
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Calculate union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
        
    return inter_area / union_area