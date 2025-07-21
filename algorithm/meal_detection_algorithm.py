import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MealDetectionAlgorithm:
    """
    A comprehensive algorithm to detect meal times and estimate calories from CGM data.
    
    Features:
    - Glucose rate of change analysis
    - Peak detection and classification
    - Meal timing estimation
    - Calorie estimation based on glucose response
    - Meal type classification (breakfast, lunch, dinner, snack)
    """
    
    def __init__(self, 
                 glucose_threshold=15,  # mg/dL increase threshold
                 time_window_minutes=180,  # 3-hour window for meal analysis
                 min_peak_height=20,  # minimum glucose increase for meal
                 min_peak_distance_minutes=60,  # minimum time between meals
                 baseline_window_minutes=30):  # window for baseline calculation
        
        self.glucose_threshold = glucose_threshold
        self.time_window_minutes = time_window_minutes
        self.min_peak_height = min_peak_height
        self.min_peak_distance_minutes = min_peak_distance_minutes
        self.baseline_window_minutes = baseline_window_minutes
        
        # Calorie estimation parameters (based on research)
        self.cal_per_mg_dl_increase = 2.5  # calories per mg/dL glucose increase
        self.baseline_calories = {
            'breakfast': 400,
            'lunch': 600,
            'dinner': 500,
            'snack': 150
        }
        
    def preprocess_data(self, df):
        """
        Preprocess the CGM data for analysis.
        """
        # Convert timestamp to datetime - handle mixed formats
        def convert_timestamp(ts):
            if pd.isna(ts):
                return pd.NaT
            ts_str = str(ts)
            if len(ts_str) == 10:  # Date only
                return pd.to_datetime(ts_str + ' 12:00:00')  # Assume noon
            else:
                return pd.to_datetime(ts_str)
        
        df['timestamp'] = df['timestamp'].apply(convert_timestamp)
        df['date'] = pd.to_datetime(df['date'])
        
        # Remove rows with invalid timestamps
        df = df[df['timestamp'].notna()]
        
        # Sort by participant and timestamp
        df = df.sort_values(['participant_id', 'timestamp'])
        
        # Remove duplicates and invalid values
        df = df.drop_duplicates(subset=['participant_id', 'timestamp'])
        df = df[df['glucose_value'].notna() & (df['glucose_value'] > 0)]
        
        return df
    
    def calculate_glucose_derivatives(self, glucose_series, timestamps):
        """
        Calculate first and second derivatives of glucose.
        """
        # First derivative (rate of change)
        glucose_diff = np.diff(glucose_series)
        
        # Handle timestamps - convert to datetime if needed
        if isinstance(timestamps.iloc[0], str):
            timestamps = pd.to_datetime(timestamps)
        
        time_diff = np.diff([ts.timestamp() for ts in timestamps])
        first_derivative = glucose_diff / time_diff * 60  # mg/dL per minute
        
        # Second derivative (acceleration)
        second_derivative = np.diff(first_derivative)
        
        return first_derivative, second_derivative
    
    def detect_glucose_peaks(self, glucose_series, timestamps, participant_id):
        """
        Detect significant glucose peaks that could indicate meals.
        """
        # Calculate moving average for baseline
        window_size = int(self.baseline_window_minutes / 5)  # 5-minute intervals
        baseline = glucose_series.rolling(window=window_size, center=True).mean()
        
        # Calculate rate of change
        first_derivative, second_derivative = self.calculate_glucose_derivatives(glucose_series, timestamps)
        
        # Find peaks using scipy - use simpler criteria for initial detection
        peaks, properties = signal.find_peaks(
            glucose_series,
            height=glucose_series.mean() + self.min_peak_height,  # Use mean instead of baseline
            distance=int(self.min_peak_distance_minutes / 5),  # minimum distance between peaks
            prominence=self.min_peak_height
        )
        
        # Filter peaks based on rate of change
        meal_peaks = []
        for peak_idx in peaks:
            if peak_idx < len(first_derivative):
                # Check if there's a significant positive rate of change before the peak
                start_idx = max(0, peak_idx - 6)  # 30 minutes before peak
                end_idx = peak_idx
                
                if end_idx < len(first_derivative):
                    avg_rate = np.mean(first_derivative[start_idx:end_idx])
                    
                    if avg_rate > 0.5:  # mg/dL per minute threshold
                        meal_peaks.append({
                            'peak_idx': peak_idx,
                            'peak_time': timestamps.iloc[peak_idx],
                            'peak_glucose': glucose_series.iloc[peak_idx],
                            'baseline_glucose': baseline.iloc[peak_idx] if not pd.isna(baseline.iloc[peak_idx]) else glucose_series.iloc[peak_idx],
                            'glucose_increase': glucose_series.iloc[peak_idx] - (baseline.iloc[peak_idx] if not pd.isna(baseline.iloc[peak_idx]) else glucose_series.iloc[peak_idx]),
                            'rate_of_change': avg_rate,
                            'participant_id': participant_id
                        })
        
        return meal_peaks
    
    def estimate_meal_time(self, peak_data, glucose_series, timestamps):
        """
        Estimate the actual meal time based on glucose response patterns.
        Typically, glucose starts rising 10-15 minutes after eating.
        """
        peak_idx = peak_data['peak_idx']
        
        # Look backwards from peak to find when glucose started rising
        start_idx = max(0, peak_idx - 12)  # 60 minutes before peak
        
        # Calculate rate of change in the window before the peak
        if peak_idx < len(glucose_series) - 1:
            window_glucose = glucose_series.iloc[start_idx:peak_idx+1]
            window_times = timestamps.iloc[start_idx:peak_idx+1]
            
            if len(window_glucose) > 1:
                # Find the point where glucose starts consistently rising
                glucose_diff = np.diff(window_glucose)
                rising_points = np.where(glucose_diff > 0.5)[0]  # 0.5 mg/dL per 5-min interval
                
                if len(rising_points) > 0:
                    # Estimate meal time as 15 minutes before glucose starts rising
                    meal_start_idx = start_idx + rising_points[0]
                    estimated_meal_time = timestamps.iloc[meal_start_idx] - timedelta(minutes=15)
                else:
                    # Fallback: estimate meal time as 15 minutes before peak
                    estimated_meal_time = timestamps.iloc[peak_idx] - timedelta(minutes=15)
            else:
                estimated_meal_time = timestamps.iloc[peak_idx] - timedelta(minutes=15)
        else:
            estimated_meal_time = timestamps.iloc[peak_idx] - timedelta(minutes=15)
        
        return estimated_meal_time
    
    def classify_meal_type(self, meal_time):
        """
        Classify meal type based on time of day.
        """
        hour = meal_time.hour
        
        if 5 <= hour < 11:
            return 'breakfast'
        elif 11 <= hour < 16:
            return 'lunch'
        elif 16 <= hour < 21:
            return 'dinner'
        else:
            return 'snack'
    
    def estimate_calories(self, peak_data, meal_type):
        """
        Estimate calories based on glucose response magnitude and meal type.
        """
        glucose_increase = peak_data['glucose_increase']
        rate_of_change = peak_data['rate_of_change']
        
        # Base calories from glucose response
        base_calories = glucose_increase * self.cal_per_mg_dl_increase
        
        # Adjust based on rate of change (faster rise = more carbs)
        rate_multiplier = min(2.0, max(0.5, rate_of_change / 1.0))
        
        # Meal type adjustment
        meal_multiplier = {
            'breakfast': 1.0,
            'lunch': 1.2,
            'dinner': 1.1,
            'snack': 0.6
        }
        
        estimated_calories = base_calories * rate_multiplier * meal_multiplier[meal_type]
        
        # Add baseline calories for the meal type
        estimated_calories += self.baseline_calories[meal_type]
        
        # Apply reasonable bounds
        estimated_calories = max(50, min(1500, estimated_calories))
        
        return round(estimated_calories)
    
    def analyze_participant(self, participant_data):
        """
        Analyze a single participant's data for meal detection.
        """
        participant_data = participant_data.sort_values('timestamp')
        
        # Detect peaks
        meal_peaks = self.detect_glucose_peaks(
            participant_data['glucose_value'],
            participant_data['timestamp'],
            participant_data['participant_id'].iloc[0]
        )
        
        meals = []
        for peak in meal_peaks:
            # Estimate meal time
            meal_time = self.estimate_meal_time(peak, participant_data['glucose_value'], participant_data['timestamp'])
            
            # Classify meal type
            meal_type = self.classify_meal_type(meal_time)
            
            # Estimate calories
            estimated_calories = self.estimate_calories(peak, meal_type)
            
            meals.append({
                'participant_id': peak['participant_id'],
                'meal_time': meal_time,
                'peak_time': peak['peak_time'],
                'meal_type': meal_type,
                'glucose_increase': peak['glucose_increase'],
                'peak_glucose': peak['peak_glucose'],
                'baseline_glucose': peak['baseline_glucose'],
                'rate_of_change': peak['rate_of_change'],
                'estimated_calories': estimated_calories
            })
        
        return meals
    
    def detect_meals(self, df):
        """
        Main method to detect meals for all participants.
        """
        print("Preprocessing data...")
        df = self.preprocess_data(df)
        
        unique_participants = df['participant_id'].unique()
        total_participants = len(unique_participants)
        print(f"Analyzing {total_participants} participants...")
        
        all_meals = []
        processed_count = 0
        successful_count = 0
        
        for i, participant_id in enumerate(unique_participants):
            processed_count += 1
            
            # Show progress every 50 participants or at the end
            if processed_count % 50 == 0 or processed_count == total_participants:
                print(f"Progress: {processed_count}/{total_participants} participants processed ({processed_count/total_participants*100:.1f}%) - {len(all_meals)} meals detected so far")
            
            participant_data = df[df['participant_id'] == participant_id]
            
            if len(participant_data) > 50:  # Minimum data requirement
                try:
                    meals = self.analyze_participant(participant_data)
                    all_meals.extend(meals)
                    successful_count += 1
                except Exception as e:
                    print(f"Error processing participant {participant_id}: {e}")
                    continue
        
        print(f"\nProcessing complete: {successful_count}/{total_participants} participants successfully analyzed")
        
        meals_df = pd.DataFrame(all_meals)
        
        if len(meals_df) > 0:
            meals_df = meals_df.sort_values(['participant_id', 'meal_time'])
        
        return meals_df
    
    def generate_summary_stats(self, meals_df):
        """
        Generate summary statistics for detected meals.
        """
        if len(meals_df) == 0:
            return "No meals detected."
        
        summary = {
            'total_meals_detected': len(meals_df),
            'unique_participants': meals_df['participant_id'].nunique(),
            'avg_calories_per_meal': meals_df['estimated_calories'].mean(),
            'avg_glucose_increase': meals_df['glucose_increase'].mean(),
            'meals_per_participant': len(meals_df) / meals_df['participant_id'].nunique()
        }
        
        # Meal type distribution
        meal_type_counts = meals_df['meal_type'].value_counts()
        summary['meal_type_distribution'] = meal_type_counts.to_dict()
        
        # Time distribution
        meals_df['meal_hour'] = meals_df['meal_time'].dt.hour
        hour_distribution = meals_df['meal_hour'].value_counts().sort_index()
        summary['hour_distribution'] = hour_distribution.to_dict()
        
        return summary
    
    def plot_meal_detection_example(self, df, participant_id, days=3):
        """
        Plot an example of meal detection for a specific participant.
        """
        participant_data = df[df['participant_id'] == participant_id].copy()
        participant_data['timestamp'] = pd.to_datetime(participant_data['timestamp'])
        participant_data = participant_data.sort_values('timestamp')
        
        # Get last N days of data
        end_date = participant_data['timestamp'].max()
        start_date = end_date - timedelta(days=days)
        plot_data = participant_data[participant_data['timestamp'] >= start_date]
        
        # Detect meals for this period
        meals = self.analyze_participant(plot_data)
        meals = [m for m in meals if m['meal_time'] >= start_date]
        
        # Create plot
        plt.figure(figsize=(15, 8))
        plt.plot(plot_data['timestamp'], plot_data['glucose_value'], 'b-', alpha=0.7, label='Glucose')
        
        # Mark detected meals
        colors = {'breakfast': 'orange', 'lunch': 'red', 'dinner': 'purple', 'snack': 'green'}
        for meal in meals:
            plt.axvline(x=meal['meal_time'], color=colors[meal['meal_type']], 
                       linestyle='--', alpha=0.8, 
                       label=f"{meal['meal_type']} ({meal['estimated_calories']} cal)")
        
        plt.title(f'Meal Detection Example - Participant {participant_id}')
        plt.xlabel('Time')
        plt.ylabel('Glucose (mg/dL)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return meals

# Example usage and testing
if __name__ == "__main__":
    # Load data
    print("Loading CGM data...")
    df = pd.read_csv('aireadi_cgm_v2.csv')
    
    # Initialize algorithm
    meal_detector = MealDetectionAlgorithm()
    
    # Detect meals
    meals_df = meal_detector.detect_meals(df)
    
    # Save results
    meals_df.to_csv('detected_meals.csv', index=False)
    print(f"Detected {len(meals_df)} meals across {meals_df['participant_id'].nunique()} participants")
    
    # Generate summary
    summary = meal_detector.generate_summary_stats(meals_df)
    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Plot example
    if len(meals_df) > 0:
        sample_participant = meals_df['participant_id'].iloc[0]
        meal_detector.plot_meal_detection_example(df, sample_participant) 