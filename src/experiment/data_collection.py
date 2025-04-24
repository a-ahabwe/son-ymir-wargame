import json
import os
import time
import numpy as np
from datetime import datetime

class ExperimentLogger:
    """Logs experiment data for analysis"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.session_data = {
            'participant_id': None,
            'condition': None,
            'start_time': None,
            'end_time': None,
            'actions': [],
            'veto_requests': [],
            'veto_decisions': [],
            'questionnaire': None
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def start_session(self, participant_id, condition):
        """Start a new experimental session"""
        # Save previous session if exists
        if self.session_data['participant_id'] is not None:
            self.end_session()
            
        # Initialize new session
        self.session_data = {
            'participant_id': participant_id,
            'condition': condition,
            'start_time': time.time(),
            'end_time': None,
            'actions': [],
            'veto_requests': [],
            'veto_decisions': [],
            'questionnaire': None
        }
        
        print(f"Started session for participant {participant_id}, condition {condition}")
        
    def log_action(self, action_data):
        """Log an action and its outcome"""
        # Add timestamp
        action_entry = {
            'timestamp': time.time(),
            **action_data
        }
        
        self.session_data['actions'].append(action_entry)
        
    def log_veto_request(self, veto_data):
        """Log a veto request"""
        # Add timestamp
        veto_entry = {
            'timestamp': time.time(),
            'response_time': None,  # Will be filled when decision is logged
            **veto_data
        }
        
        request_id = len(self.session_data['veto_requests'])
        veto_entry['request_id'] = request_id
        
        self.session_data['veto_requests'].append(veto_entry)
        return request_id
        
    def log_veto(self, veto_data):
        """Log a veto decision"""
        # Calculate response time if possible
        if 'request_id' in veto_data and veto_data['request_id'] < len(self.session_data['veto_requests']):
            request = self.session_data['veto_requests'][veto_data['request_id']]
            response_time = time.time() - request['timestamp']
            veto_data['response_time'] = response_time
            
            # Update the request with response time
            request['response_time'] = response_time
            
        # Add timestamp
        veto_entry = {
            'timestamp': time.time(),
            **veto_data
        }
        
        self.session_data['veto_decisions'].append(veto_entry)
        
    def log_questionnaire(self, questionnaire_data):
        """Log subjective measures from questionnaires"""
        self.session_data['questionnaire'] = questionnaire_data
        
    def end_session(self):
        """End session and save data"""
        if self.session_data['participant_id'] is None:
            return  # No active session
            
        # Set end time
        self.session_data['end_time'] = time.time()
        
        # Calculate session duration
        duration = self.session_data['end_time'] - self.session_data['start_time']
        self.session_data['duration'] = duration
        
        # Convert numpy arrays and other non-serializable types to regular Python types
        for action in self.session_data['actions']:
            for key, value in list(action.items()):
                # Handle numpy arrays
                if hasattr(value, 'tolist'):
                    action[key] = value.tolist()
                # Handle numpy scalars
                elif hasattr(value, 'item'):
                    action[key] = value.item()
                    
        # Convert in veto decisions and requests as well
        for veto in self.session_data['veto_decisions']:
            for key, value in list(veto.items()):
                if hasattr(value, 'tolist'):
                    veto[key] = value.tolist()
                elif hasattr(value, 'item'):
                    veto[key] = value.item()
                    
        for request in self.session_data['veto_requests']:
            for key, value in list(request.items()):
                if hasattr(value, 'tolist'):
                    request[key] = value.tolist()
                elif hasattr(value, 'item'):
                    request[key] = value.item()
        
        # Save to file
        filename = self._get_session_filename()
        with open(filename, 'w') as f:
            json.dump(self.session_data, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
            
        print(f"Session ended. Data saved to {filename}")
        
        # Reset session data
        participant_id = self.session_data['participant_id']
        self.session_data = {
            'participant_id': None,
            'condition': None,
            'start_time': None,
            'end_time': None,
            'actions': [],
            'veto_requests': [],
            'veto_decisions': [],
            'questionnaire': None
        }
        
        return participant_id
        
    def _get_session_filename(self):
        """Generate a filename for the session data"""
        participant_id = self.session_data['participant_id']
        condition = self.session_data['condition']
        timestamp = datetime.fromtimestamp(self.session_data['start_time']).strftime("%Y%m%d_%H%M%S")
        
        return f"{self.output_dir}/p{participant_id}_{condition}_{timestamp}.json"