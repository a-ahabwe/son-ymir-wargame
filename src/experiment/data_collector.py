from src.experiment.data_collection import ExperimentLogger

class DataCollector:
    """Wrapper class for ExperimentLogger to maintain compatibility with game code"""
    def __init__(self, participant_id, log_dir):
        self.logger = ExperimentLogger(log_dir)
        self.logger.start_session(participant_id, 'default')
        
    def log_veto(self, veto_data):
        """Log a veto decision"""
        self.logger.log_veto(veto_data)
        
    def log_veto_request(self, request_data):
        """Log a veto request"""
        return self.logger.log_veto_request(request_data)
        
    def log_action(self, **action_data):
        """Log a game action with any provided data"""
        self.logger.log_action(action_data)
        
    def log_veto_event(self, **veto_info):
        """Log a veto event from the game"""
        self.logger.log_veto(veto_info)
        
    def save(self):
        """Save all logged data"""
        self.logger.end_session() 