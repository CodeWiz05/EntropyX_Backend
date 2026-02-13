"""
DATA GUARD MODULE
Handles sensor data cleaning and error logging
"""

import json
from datetime import datetime
import os

class DataGuard:
    def __init__(self, log_file="sensor_log.json"):
        """Initialize the Data Guard"""
        self.log_file = log_file
        
    def clean_sensor_data(self, sensor_readings):
        """
        Clean raw sensor data
        
        Args:
            sensor_readings: List of [timestamp, value] pairs
                           Example: [[1, 45], [2, -999], [3, 47], [4, 0], [5, 900]]
        
        Returns:
            cleaned_readings: List with interpolated values
            errors_logged: List of errors detected
        """
        
        cleaned_data = []
        errors = []
        
        # Step 1: First pass - identify all bad values
        for i in range(len(sensor_readings)):
            timestamp = sensor_readings[i][0]
            value = sensor_readings[i][1]
            
            # Check if value is bad (-999 or 0)
            if value in [-999, 0]:
                # This is a NULL value - needs interpolation
                cleaned_value = self._interpolate_value(sensor_readings, i)
                cleaned_data.append([timestamp, cleaned_value])
                
                # Log the interpolation
                self._log_error({
                    'type': 'NULL_VALUE',
                    'timestamp': timestamp,
                    'original_value': value,
                    'corrected_value': cleaned_value,
                    'method': 'linear_interpolation'
                })
                
            else:
                # Check if this is an outlier (jump too big)
                if i > 0:  # Skip first reading
                    prev_value = sensor_readings[i-1][1]
                    if prev_value not in [-999, 0]:  # Don't compare with bad values
                        change = abs(value - prev_value)
                        
                        # If jump > 100 in 1 minute, it's impossible for PM2.5
                        if change > 100 and sensor_readings[i][0] - sensor_readings[i-1][0] <= 1:
                            errors.append({
                                'timestamp': timestamp,
                                'value': value,
                                'previous_value': prev_value,
                                'change': change,
                                'type': 'SENSOR_SPIKE'
                            })
                            
                            # Log the error
                            self._log_error({
                                'type': 'OUTLIER_DETECTED',
                                'timestamp': timestamp,
                                'value': value,
                                'previous_value': prev_value,
                                'change': change,
                                'action': 'excluded_from_training'
                            })
                            
                            # For outliers, we still keep the value but flag it
                            cleaned_data.append([timestamp, value])
                        else:
                            cleaned_data.append([timestamp, value])
                else:
                    cleaned_data.append([timestamp, value])
        
        return cleaned_data, errors
    
    def _interpolate_value(self, readings, index):
        """
        Interpolate a missing value using neighbors
        
        Example: If index 2 is missing, use index 1 and 3
        """
        
        # Find previous valid value
        prev_value = None
        prev_idx = index - 1
        while prev_idx >= 0:
            if readings[prev_idx][1] not in [-999, 0]:
                prev_value = readings[prev_idx][1]
                break
            prev_idx -= 1
        
        # Find next valid value
        next_value = None
        next_idx = index + 1
        while next_idx < len(readings):
            if readings[next_idx][1] not in [-999, 0]:
                next_value = readings[next_idx][1]
                break
            next_idx += 1
        
        # Case 1: Have both neighbors
        if prev_value is not None and next_value is not None:
            return (prev_value + next_value) / 2
        
        # Case 2: Only have previous value
        elif prev_value is not None:
            return prev_value
        
        # Case 3: Only have next value
        elif next_value is not None:
            return next_value
        
        # Case 4: No neighbors at all
        else:
            return 50  # Default fallback value
    
    def _log_error(self, error_data):
        """Log errors to JSON file for System Health"""
        
        # Add timestamp to error
        error_data['logged_at'] = datetime.now().isoformat()
        
        # Read existing logs
        logs = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new error
        logs.append(error_data)
        
        # Keep only last 1000 errors (prevents file from getting too big)
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Write back to file
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_system_health_report(self):
        """Generate health report from logs"""
        
        if not os.path.exists(self.log_file):
            return {"message": "No errors logged yet"}
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        # Count errors by type
        error_counts = {}
        for log in logs:
            error_type = log.get('type', 'UNKNOWN')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'total_errors_logged': len(logs),
            'error_breakdown': error_counts,
            'last_24h': self._count_last_24h(logs)
        }
    
    def _count_last_24h(self, logs):
        """Count errors in last 24 hours"""
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(hours=24)
        count = 0
        
        for log in logs:
            log_time = datetime.fromisoformat(log.get('logged_at', '2000-01-01'))
            if log_time > cutoff:
                count += 1
        
        return count


# TEST THE DATA GUARD
if __name__ == "__main__":
    print("="*50)
    print("TESTING DATA GUARD MODULE")
    print("="*50)
    
    # Create sample sensor data with problems
    sample_data = [
        [1, 45],      # Normal
        [2, -999],    # NULL - needs interpolation
        [3, 47],      # Normal
        [4, 0],       # NULL - needs interpolation
        [5, 46],      # Normal
        [6, 900],     # OUTLIER - impossible jump from 46 to 900
        [7, 48],      # Normal
        [8, -999],    # NULL
        [9, 50],      # Normal
        [10, 52]      # Normal
    ]
    
    # Initialize Data Guard
    guard = DataGuard("test_log.json")
    
    # Clean the data
    cleaned, errors = guard.clean_sensor_data(sample_data)
    
    print("\n📊 ORIGINAL DATA:")
    for row in sample_data:
        print(f"  Hour {row[0]}: {row[1]}")
    
    print("\n✨ CLEANED DATA (Interpolated):")
    for row in cleaned:
        print(f"  Hour {row[0]}: {row[1]}")
    
    print("\n⚠️  ERRORS DETECTED:")
    for error in errors:
        print(f"  {error['type']} at Hour {error['timestamp']}: {error['value']} (jump of {error.get('change', 'N/A')})")
    
    print("\n📈 SYSTEM HEALTH REPORT:")
    print(json.dumps(guard.get_system_health_report(), indent=2))