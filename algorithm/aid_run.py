import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_palette("husl")

class AutomatedInsulinController:
    def __init__(self, target_range=(70, 180), prediction_horizon=60):
        """
        Automated Insulin Delivery Controller - FIXED VERSION with Visualizations
        
        Parameters:
        - target_range: tuple (low, high) for target glucose range in mg/dL
        - prediction_horizon: minutes ahead to predict glucose
        """
        self.target_low = target_range[0]
        self.target_high = target_range[1]
        self.target_center = (target_range[0] + target_range[1]) / 2
        self.prediction_horizon = prediction_horizon
        
        # FIXED: More reasonable safety constraints
        self.max_bolus = 4.0  # Maximum single bolus in units (reduced from 8.0)
        self.max_hourly_rate = 10.0  # Maximum insulin delivery per hour (reduced from 15.0)
        self.min_glucose_threshold = 70  # Don't give insulin below target (was 65)
        
        # FIXED: More realistic physiological parameters
        self.insulin_sensitivity = 40  # mg/dL per unit (more conservative)
        self.carb_ratio = 12  # grams carb per unit insulin
        self.insulin_duration = 180  # minutes (3 hours, more realistic)
        self.insulin_peak = 60  # minutes to peak action (faster peak)
        
        # Models
        self.glucose_predictor = None
        self.scaler = StandardScaler()
        
        # History tracking
        self.glucose_history = []
        self.insulin_history = []
        self.food_history = []
        self.insulin_on_board = 0
        
        # ADDED: Glucose state tracking for simulation
        self.glucose_state = None  # Will store modified glucose values during simulation
        self.original_cgm_data = None  # Store original data for reference
        
        # ADDED: Control parameters
        self.control_aggressiveness = 0.5  # How aggressive the control is (0-1) - increased
        
    def insulin_action_curve(self, time_minutes):
        """
        FIXED: More realistic insulin action curve
        """
        if time_minutes <= 0:
            return 0.0  # No immediate action
        elif time_minutes >= self.insulin_duration:
            return 0.0
        else:
            # More realistic curve: slow onset, peak at 60 min, gradual decay
            if time_minutes <= self.insulin_peak:
                # Gradual onset
                return (time_minutes / self.insulin_peak) * 0.8
            else:
                # Exponential decay from peak
                decay_rate = 0.015
                peak_action = 0.8
                return peak_action * np.exp(-decay_rate * (time_minutes - self.insulin_peak))
    
    def update_insulin_on_board(self, current_time):
        """Update insulin on board based on previous deliveries"""
        self.insulin_on_board = 0
        current_time_minutes = current_time
        
        for delivery_time, insulin_amount in self.insulin_history[-20:]:  # Only last 20 doses
            time_since_delivery = current_time_minutes - delivery_time
            if 0 < time_since_delivery < self.insulin_duration:
                remaining_action = self.insulin_action_curve(time_since_delivery)
                self.insulin_on_board += insulin_amount * remaining_action
    
    def create_features_from_state(self, glucose_data, food_data, current_idx):
        """
        Create features from glucose state (modified glucose values)
        """
        features = []
        
        # Current glucose
        if current_idx >= 0 and current_idx < len(glucose_data):
            if isinstance(glucose_data, pd.Series):
                features.append(glucose_data.iloc[current_idx])
            else:
                features.append(glucose_data[current_idx])
        else:
            features.append(100)
            
        # Glucose trends (improved)
        for lookback in [1, 3, 6, 12]:  # 5, 15, 30, 60 minutes
            if current_idx >= lookback:
                if isinstance(glucose_data, pd.Series):
                    past_glucose = glucose_data.iloc[current_idx - lookback]
                    current_val = glucose_data.iloc[current_idx] if current_idx < len(glucose_data) else 100
                else:
                    past_glucose = glucose_data[current_idx - lookback]
                    current_val = glucose_data[current_idx] if current_idx < len(glucose_data) else 100
                trend = current_val - past_glucose
                features.append(trend)
            else:
                features.append(0)
        
        # Rate of change (more stable)
        if current_idx >= 2:
            if isinstance(glucose_data, pd.Series):
                rate = (glucose_data.iloc[current_idx] - glucose_data.iloc[current_idx - 2]) / 2
            else:
                rate = (glucose_data[current_idx] - glucose_data[current_idx - 2]) / 2
            features.append(rate)
        else:
            features.append(0)
            
        # Food intake
        if current_idx >= 0 and current_idx < len(food_data):
            features.append(food_data.iloc[current_idx])
        else:
            features.append(0)
            
        # Time-based features
        hour_of_day = (current_idx * 5 / 60) % 24
        features.extend([
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24)
        ])
        
        # Insulin on board
        features.append(self.insulin_on_board)
        
        return np.array(features)

    def create_features(self, cgm_data, food_data, current_idx):
        """
        IMPROVED: Create better feature vector for glucose prediction
        """
        features = []
        
        # Current glucose
        if current_idx >= 0 and current_idx < len(cgm_data):
            features.append(cgm_data.iloc[current_idx])
        else:
            features.append(100)
            
        # Glucose trends (improved)
        for lookback in [1, 3, 6, 12]:  # 5, 15, 30, 60 minutes
            if current_idx >= lookback:
                past_glucose = cgm_data.iloc[current_idx - lookback]
                current_val = cgm_data.iloc[current_idx] if current_idx < len(cgm_data) else 100
                trend = current_val - past_glucose
                features.append(trend)
            else:
                features.append(0)
        
        # Rate of change (more stable)
        if current_idx >= 2:
            rate = (cgm_data.iloc[current_idx] - cgm_data.iloc[current_idx - 2]) / 2
            features.append(rate)
        else:
            features.append(0)
            
        # Food intake
        if current_idx >= 0 and current_idx < len(food_data):
            features.append(food_data.iloc[current_idx])
        else:
            features.append(0)
            
        # Time-based features
        hour_of_day = (current_idx * 5 / 60) % 24
        features.extend([
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24)
        ])
        
        # Insulin on board
        features.append(self.insulin_on_board)
        
        return np.array(features)
    
    def train_glucose_predictor(self, cgm_data, food_data):
        """
        IMPROVED: Better training approach
        """
        X, y = [], []
        prediction_steps = max(1, self.prediction_horizon // 5)
        
        # Create training data
        for i in range(len(cgm_data) - prediction_steps):
            if i + prediction_steps < len(cgm_data):
                features = self.create_features(cgm_data, food_data, i)
                X.append(features)
                y.append(cgm_data.iloc[i + prediction_steps])
        
        if len(X) < 50:  # Need more data for reliable training
            return False
            
        X = np.array(X)
        y = np.array(y)
        
        # Clean data
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        if len(X) < 50:
            return False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # IMPROVED: Better model parameters
        self.glucose_predictor = RandomForestRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=8,      # Prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.glucose_predictor.fit(X_scaled, y)
        
        return True
    
    def predict_glucose_with_insulin(self, cgm_data, food_data, current_idx, insulin_dose):
        """
        FIXED: Properly predict glucose considering insulin timing
        """
        if current_idx < 0 or current_idx >= len(cgm_data):
            return 100
            
        # Use glucose state if available, otherwise original data
        if self.glucose_state is not None and current_idx < len(self.glucose_state):
            current_glucose = self.glucose_state[current_idx]
        else:
            current_glucose = cgm_data.iloc[current_idx]
        
        # Base prediction without additional insulin
        if self.glucose_predictor is not None:
            # Use glucose state for feature creation if available
            if self.glucose_state is not None:
                features = self.create_features_from_state(self.glucose_state, food_data, current_idx)
            else:
                features = self.create_features(cgm_data, food_data, current_idx)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            predicted_glucose = self.glucose_predictor.predict(features_scaled)[0]
        else:
            # Simple trend-based prediction using glucose state
            if current_idx >= 2:
                if self.glucose_state is not None:
                    recent_trend = self.glucose_state[current_idx] - self.glucose_state[current_idx-2]
                else:
                    recent_trend = cgm_data.iloc[current_idx] - cgm_data.iloc[current_idx-2]
                predicted_glucose = current_glucose + recent_trend * (self.prediction_horizon / 10)
            else:
                predicted_glucose = current_glucose
        
        # FIXED: Apply insulin effect with proper timing
        if insulin_dose > 0:
            # Insulin effect peaks at prediction horizon
            time_to_prediction = self.prediction_horizon
            insulin_action = self.insulin_action_curve(time_to_prediction)
            insulin_effect = insulin_dose * self.insulin_sensitivity * insulin_action
            predicted_glucose -= insulin_effect
        
        return predicted_glucose
    
    def calculate_optimal_insulin(self, cgm_data, food_data, current_idx):
        """
        IMPROVED: Enhanced insulin calculation with trend consideration
        """
        if current_idx < 0 or current_idx >= len(cgm_data):
            return 0
            
        current_glucose = cgm_data.iloc[current_idx]
        
        # Calculate glucose trend
        glucose_trend = 0
        if current_idx >= 2:
            glucose_trend = (cgm_data.iloc[current_idx] - cgm_data.iloc[current_idx-2]) / 2  # mg/dL per 5min
        
        # Predict glucose without insulin
        predicted_glucose = self.predict_glucose_with_insulin(cgm_data, food_data, current_idx, 0)
        
        # Enhanced decision making
        target_glucose = self.target_center
        
        # Don't give insulin if:
        # 1. Current glucose is in target and trending down
        # 2. Predicted glucose is in target
        if (current_glucose <= self.target_high and glucose_trend <= -2) or predicted_glucose <= self.target_high:
            return 0
            
        # Calculate glucose excess considering both current and predicted
        current_excess = max(0, current_glucose - target_glucose)
        predicted_excess = max(0, predicted_glucose - target_glucose)
        
        # Weight current vs predicted based on trend
        if glucose_trend > 5:  # Rising rapidly
            glucose_excess = 0.3 * current_excess + 0.7 * predicted_excess
        elif glucose_trend > 0:  # Rising slowly
            glucose_excess = 0.5 * current_excess + 0.5 * predicted_excess  
        else:  # Stable or falling
            glucose_excess = 0.7 * current_excess + 0.3 * predicted_excess
        
        if glucose_excess <= 0:
            return 0
            
        # Calculate insulin needed
        desired_reduction = glucose_excess * self.control_aggressiveness
        
        # Account for existing IOB effect
        iob_effect = self.insulin_on_board * self.insulin_sensitivity * 0.6
        net_reduction_needed = max(0, desired_reduction - iob_effect)
        
        # Convert to insulin dose with safety factor
        insulin_dose = net_reduction_needed / (self.insulin_sensitivity * 0.8)
        
        return max(0, insulin_dose)
    
    def apply_insulin_effect(self, current_glucose, current_time):
        """
        ADDED: Apply cumulative insulin effects to current glucose
        """
        total_insulin_effect = 0
        
        # Calculate cumulative effect from all previous insulin deliveries
        for delivery_time, insulin_amount in self.insulin_history:
            time_since_delivery = current_time - delivery_time
            if 0 < time_since_delivery < self.insulin_duration:
                # Get the insulin action at this time point
                action_fraction = self.insulin_action_curve(time_since_delivery)
                # Calculate glucose reduction from this insulin dose
                insulin_effect = insulin_amount * self.insulin_sensitivity * action_fraction
                total_insulin_effect += insulin_effect
        
        # Apply the effect (reduce glucose)
        modified_glucose = current_glucose - total_insulin_effect
        
        # Ensure glucose doesn't go negative
        return max(modified_glucose, 20)  # Minimum glucose of 20 mg/dL
    
    def safety_check(self, current_glucose, proposed_insulin):
        """
        IMPROVED: Better safety constraints
        """
        # Never give insulin if glucose is at or below target low
        if current_glucose <= self.target_low:
            return 0
        
        # Be very conservative if glucose is near target
        if current_glucose <= self.target_high + 10:
            proposed_insulin *= 0.5
        
        # Apply dosing limits
        proposed_insulin = min(proposed_insulin, self.max_bolus)
        proposed_insulin = min(proposed_insulin, self.max_hourly_rate / 12)  # Per 5-min interval
        
        return max(0, proposed_insulin)
    
    def calculate_insulin_dose(self, cgm_data, food_data, current_idx, current_time):
        """
        FIXED: Main control logic
        """
        if current_idx < 0 or current_idx >= len(cgm_data):
            return 0
        
        # Update insulin on board
        self.update_insulin_on_board(current_time)
        
        # Calculate optimal dose
        optimal_insulin = self.calculate_optimal_insulin(cgm_data, food_data, current_idx)
        
        # Apply safety checks
        safe_insulin = self.safety_check(cgm_data.iloc[current_idx], optimal_insulin)
        
        # Only record significant doses
        if safe_insulin > 0.05:
            self.insulin_history.append((current_time, safe_insulin))
            # Limit history size
            if len(self.insulin_history) > 100:
                self.insulin_history = self.insulin_history[-50:]
        
        return safe_insulin
    
    def personalize_parameters(self, cgm_data, food_data):
        """
        IMPROVED: Better parameter personalization
        """
        glucose_mean = cgm_data.mean()
        glucose_std = cgm_data.std()
        food_mean = food_data.mean() if len(food_data) > 0 else 100
        
        # Base personalization on glucose control quality
        baseline_tir = ((cgm_data >= self.target_low) & (cgm_data <= self.target_high)).mean() * 100
        
        # Adjust based on current control quality
        if baseline_tir < 60:  # Poor control - be more aggressive
            self.insulin_sensitivity = 35
            self.control_aggressiveness = 0.7
        elif baseline_tir < 70:  # Moderate control
            self.insulin_sensitivity = 40
            self.control_aggressiveness = 0.5
        elif baseline_tir < 80:  # Good control
            self.insulin_sensitivity = 45
            self.control_aggressiveness = 0.3
        else:  # Excellent control - be conservative
            self.insulin_sensitivity = 50
            self.control_aggressiveness = 0.2
        
        # Adjust based on mean glucose level
        if glucose_mean > 120:
            self.control_aggressiveness *= 1.3  # More aggressive for high glucose
        elif glucose_mean > 180:
            self.control_aggressiveness *= 1.1
        
        # Adjust based on variability - more conservative if highly variable
        if glucose_std > 30:
            self.control_aggressiveness *= 0.7
        elif glucose_std < 15:
            self.control_aggressiveness *= 1.2
        
        # Safety limits
        self.control_aggressiveness = max(0.1, min(0.8, self.control_aggressiveness))
        
        print(f"  Personalized insulin sensitivity: {self.insulin_sensitivity:.1f} mg/dL/U")
        print(f"  Control aggressiveness: {self.control_aggressiveness:.2f}")
        print(f"  Baseline TIR: {baseline_tir:.1f}%")
    
    def simulate_control(self, cgm_data, food_data, subject_id=None):
        """
        FIXED: Simulation with proper glucose state tracking and insulin effects
        """
        # Reset state
        self.insulin_history = []
        self.insulin_on_board = 0
        
        # ADDED: Initialize glucose state tracking
        self.original_cgm_data = cgm_data.copy()
        self.glucose_state = cgm_data.values.copy()  # Start with original values
        
        results = {
            'timestamps': [],
            'glucose_values': [],
            'original_glucose_values': [],  # ADDED: Store original values for comparison
            'predicted_glucose': [],
            'insulin_doses': [],
            'insulin_on_board': [],
            'in_range': [],
            'time_in_range': 0,
            'hypoglycemia_time': 0,
            'hyperglycemia_time': 0,
            'control_start_time': 0  # ADDED: Store when control started
        }
        
        # Personalize parameters
        self.personalize_parameters(cgm_data, food_data)
        
        # Train prediction model
        model_trained = self.train_glucose_predictor(cgm_data, food_data)
        print(f"  Glucose prediction model trained: {model_trained}")
        
        # Simulate control
        total_points = len(cgm_data)
        in_range_count = 0
        hypo_count = 0
        hyper_count = 0
        
        # Only apply control to latter portion to allow for training
        control_start = max(50, len(cgm_data) // 5)  # Start control after 20% of data (was 25%)
        results['control_start_time'] = control_start * 5  # Store control start time in minutes
        
        for i in range(total_points):
            current_time = i * 5  # 5-minute intervals
            
            # FIXED: Use glucose state (modified by insulin) instead of original data
            current_glucose = self.glucose_state[i]
            
            # Calculate insulin dose (only after control start)
            if i >= control_start:
                insulin_dose = self.calculate_insulin_dose(cgm_data, food_data, i, current_time)
            else:
                insulin_dose = 0
                # Still update IOB for any historical insulin
                self.update_insulin_on_board(current_time)
            
            # ADDED: Apply insulin effects to glucose state for future time points
            if i < total_points - 1 and insulin_dose > 0:  # Don't modify the last point
                # Apply cumulative insulin effects to future glucose values
                for future_idx in range(i + 1, min(i + 37, total_points)):  # Next 3 hours (36 intervals)
                    future_time = future_idx * 5
                    time_diff = future_time - current_time
                    
                    if time_diff > 0 and time_diff <= self.insulin_duration:
                        # Calculate insulin effect at this future time point
                        action_fraction = self.insulin_action_curve(time_diff)
                        insulin_effect = insulin_dose * self.insulin_sensitivity * action_fraction
                        
                        # Apply the effect (reduce glucose)
                        self.glucose_state[future_idx] = max(self.glucose_state[future_idx] - insulin_effect, 20)
            
            # Predict glucose
            predicted_glucose = self.predict_glucose_with_insulin(cgm_data, food_data, i, insulin_dose)
            
            # Check if in range using modified glucose
            in_range = self.target_low <= current_glucose <= self.target_high
            if in_range:
                in_range_count += 1
            elif current_glucose < self.target_low:
                hypo_count += 1
            else:
                hyper_count += 1
            
            # Store results
            results['timestamps'].append(current_time)
            results['glucose_values'].append(current_glucose)  # Now using modified values
            results['original_glucose_values'].append(self.original_cgm_data.iloc[i])  # ADDED: Store original values
            results['predicted_glucose'].append(predicted_glucose)
            results['insulin_doses'].append(insulin_dose)
            results['insulin_on_board'].append(self.insulin_on_board)
            results['in_range'].append(in_range)
        
        # Calculate metrics using modified glucose values
        modified_glucose_series = pd.Series(self.glucose_state)
        results['time_in_range'] = (in_range_count / total_points) * 100
        results['hypoglycemia_time'] = (hypo_count / total_points) * 100
        results['hyperglycemia_time'] = (hyper_count / total_points) * 100
        results['mean_glucose'] = modified_glucose_series.mean()  # Use modified glucose
        results['glucose_std'] = modified_glucose_series.std()    # Use modified glucose
        results['total_insulin'] = sum(results['insulin_doses'])
        
        # ADDED: Control period metrics using modified glucose
        control_period_start = control_start
        if control_period_start < len(modified_glucose_series):
            control_glucose = modified_glucose_series.iloc[control_period_start:]
            control_in_range = ((control_glucose >= self.target_low) & 
                              (control_glucose <= self.target_high)).mean() * 100
            results['control_period_tir'] = control_in_range
            results['control_period_mean_glucose'] = control_glucose.mean()
            
            # Use original data for baseline comparison
            baseline_glucose = self.original_cgm_data.iloc[:control_period_start]
            if len(baseline_glucose) > 0:
                baseline_tir = ((baseline_glucose >= self.target_low) & 
                               (baseline_glucose <= self.target_high)).mean() * 100
                results['tir_improvement_control_period'] = control_in_range - baseline_tir
            else:
                results['tir_improvement_control_period'] = 0
        else:
            results['control_period_tir'] = results['time_in_range']
            results['tir_improvement_control_period'] = 0
        
        return results

# NEW: Comprehensive visualization functions

def plot_glucose_comparison(original_glucose, controlled_glucose, insulin_doses, timestamps, subject_id, target_range=(70, 180), control_start_time=None, in_range_data=None, time_in_range_pct=None):
    """
    Create a comparison plot showing glucose with and without controller
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    
    # Convert timestamps to hours for better visualization
    time_hours = np.array(timestamps) / 60
    
    # 1. Glucose comparison
    ax1 = axes[0]
    ax1.plot(time_hours, original_glucose, 'r-', linewidth=2, label='Without Controller', alpha=0.8)
    ax1.plot(time_hours, controlled_glucose, 'b-', linewidth=2, label='With Controller', alpha=0.8)
    ax1.axhline(target_range[0], color='green', linestyle=':', alpha=0.7, label=f'Target Range ({target_range[0]}-{target_range[1]} mg/dL)')
    ax1.axhline(target_range[1], color='green', linestyle=':', alpha=0.7)
    ax1.fill_between(time_hours, target_range[0], target_range[1], alpha=0.2, color='green')
    
    if control_start_time is not None:
        ax1.axvline(control_start_time/60, color='orange', linestyle='--', alpha=0.7, label='Controller Start')
    
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_title(f'Subject {subject_id} - Glucose Control Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(50, max(350, max(max(original_glucose), max(controlled_glucose)) + 20))
    
    # 2. Glucose difference
    ax2 = axes[1]
    glucose_diff = np.array(controlled_glucose) - np.array(original_glucose)
    ax2.plot(time_hours, glucose_diff, 'purple', linewidth=2, alpha=0.8)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(time_hours, glucose_diff, 0, where=(glucose_diff < 0), alpha=0.3, color='blue', label='Glucose Reduction')
    ax2.fill_between(time_hours, glucose_diff, 0, where=(glucose_diff > 0), alpha=0.3, color='red', label='Glucose Increase')
    
    if control_start_time is not None:
        ax2.axvline(control_start_time/60, color='orange', linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('Glucose Difference (mg/dL)')
    ax2.set_title('Controller Effect (Controlled - Original)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Insulin delivery
    ax3 = axes[2]
    insulin_doses = np.array(insulin_doses)
    insulin_times = time_hours[insulin_doses > 0.05]
    insulin_amounts = insulin_doses[insulin_doses > 0.05]
    
    if len(insulin_times) > 0:
        # Add smoothed trend line only
        from scipy.interpolate import UnivariateSpline
        from scipy.ndimage import gaussian_filter1d
        
        # Create a smoothed version of insulin delivery over time
        insulin_smooth = np.zeros_like(time_hours)
        for i, dose in enumerate(insulin_doses):
            if dose > 0.05:
                insulin_smooth[i] = dose
        
        # Apply Gaussian smoothing for trend
        if np.sum(insulin_smooth) > 0:
            smoothed_trend = gaussian_filter1d(insulin_smooth, sigma=5)
            ax3.plot(time_hours, smoothed_trend, 'orange', linewidth=2, alpha=0.7, label='Delivery Trend')
            ax3.legend(loc='upper right', fontsize=9)
        
        ax3.set_ylabel('Insulin Dose (U)')
        ax3.set_title('Insulin Delivery')
        ax3.grid(True, alpha=0.15)
    else:
        ax3.text(0.5, 0.5, 'No insulin delivered', transform=ax3.transAxes, 
                ha='center', va='center', fontsize=12)
        ax3.set_ylabel('Insulin Dose (U)')
        ax3.set_title('Insulin Delivery')
    
    # Set same x-axis limits as other panels
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xlabel('Time (hours)')

    # 4. Time in Range indicator
    ax4 = axes[3]
    if in_range_data is not None and time_in_range_pct is not None:
        in_range_colors = ['red' if not ir else 'green' for ir in in_range_data]
        ax4.scatter(time_hours, [1]*len(time_hours), c=in_range_colors, s=5, alpha=0.7)
        ax4.set_ylabel('In Range')
        ax4.set_xlabel('Time (hours)')
        ax4.set_title(f'Time in Range: {time_in_range_pct:.1f}%')
        ax4.set_ylim(0.5, 1.5)
        ax4.set_yticks([1])
        ax4.set_yticklabels(['TIR'])
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(ax1.get_xlim())
    else:
        ax4.text(0.5, 0.5, 'No TIR data available', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=12)
        ax4.set_ylabel('In Range')
        ax4.set_xlabel('Time (hours)')
        ax4.set_title('Time in Range')
        ax4.set_xlim(ax1.get_xlim())
    
    plt.tight_layout()
    return fig

def plot_subject_overview(results, subject_id, target_range=(70, 180)):
    """
    Create a comprehensive overview plot for a single subject
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Convert timestamps to hours for better visualization
    time_hours = np.array(results['timestamps']) / 60
    
    # 1. Glucose trace with target range
    ax1 = axes[0]
    ax1.plot(time_hours, results['glucose_values'], 'b-', linewidth=1.5, label='Actual Glucose', alpha=0.8)
    ax1.plot(time_hours, results['predicted_glucose'], 'r--', linewidth=1, label='Predicted Glucose', alpha=0.6)
    ax1.axhline(target_range[0], color='green', linestyle=':', alpha=0.7, label=f'Target Range ({target_range[0]}-{target_range[1]} mg/dL)')
    ax1.axhline(target_range[1], color='green', linestyle=':', alpha=0.7)
    ax1.fill_between(time_hours, target_range[0], target_range[1], alpha=0.2, color='green')
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_title(f'Subject {subject_id} - Glucose Control Overview')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(50, max(250, max(results['glucose_values']) + 20))
    
    # 2. Insulin delivery
    ax2 = axes[1]
    insulin_doses = np.array(results['insulin_doses'])
    insulin_times = time_hours[insulin_doses > 0.05]
    insulin_amounts = insulin_doses[insulin_doses > 0.05]
    
    if len(insulin_times) > 0:
        # Add smoothed trend line only
        from scipy.ndimage import gaussian_filter1d
        
        # Create a smoothed version of insulin delivery over time
        insulin_smooth = np.zeros_like(time_hours)
        for i, dose in enumerate(results['insulin_doses']):
            if dose > 0.05:
                insulin_smooth[i] = dose
        
        # Apply Gaussian smoothing for trend
        if np.sum(insulin_smooth) > 0:
            smoothed_trend = gaussian_filter1d(insulin_smooth, sigma=5)
            ax2.plot(time_hours, smoothed_trend, 'orange', linewidth=2, alpha=0.7, label='Delivery Trend')
            ax2.legend(loc='upper right', fontsize=9)
        
        ax2.set_ylabel('Insulin Dose (U)')
        ax2.set_title('Insulin Delivery')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No insulin delivered', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12)
        ax2.set_ylabel('Insulin Dose (U)')
        ax2.set_title('Insulin Delivery')
    
    # 3. Insulin on Board
    ax3 = axes[2]
    ax3.plot(time_hours, results['insulin_on_board'], 'orange', linewidth=2, label='Insulin on Board')
    ax3.set_ylabel('IOB (U)')
    ax3.set_title('Insulin on Board')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Time in Range indicator
    ax4 = axes[3]
    in_range_colors = ['red' if not ir else 'green' for ir in results['in_range']]
    ax4.scatter(time_hours, [1]*len(time_hours), c=in_range_colors, s=5, alpha=0.7)
    ax4.set_ylabel('In Range')
    ax4.set_xlabel('Time (hours)')
    ax4.set_title(f'Time in Range: {results["time_in_range"]:.1f}%')
    ax4.set_ylim(0.5, 1.5)
    ax4.set_yticks([1])
    ax4.set_yticklabels(['TIR'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_insulin_action_curve(controller):
    """
    Visualize the insulin action curve used by the controller
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    time_points = np.linspace(0, controller.insulin_duration, 200)
    action_values = [controller.insulin_action_curve(t) for t in time_points]
    
    ax.plot(time_points, action_values, 'b-', linewidth=2, label='Insulin Action')
    ax.axvline(controller.insulin_peak, color='red', linestyle='--', alpha=0.7, label=f'Peak at {controller.insulin_peak} min')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Insulin Action (fraction)')
    ax.set_title('Insulin Action Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_summary_statistics(summary_df):
    """
    Create summary plots across all subjects
    """
    if len(summary_df) == 0:
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # TIR Improvement
    ax1 = axes[0, 0]
    summary_df['tir_improvement'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('TIR Improvement by Subject')
    ax1.set_ylabel('TIR Improvement (%)')
    ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=45)
    
    # Total Insulin Usage
    ax2 = axes[0, 1]
    summary_df['total_insulin'].plot(kind='bar', ax=ax2, color='purple')
    ax2.set_title('Total Insulin Usage by Subject')
    ax2.set_ylabel('Total Insulin (units)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Mean Glucose
    ax3 = axes[0, 2]
    summary_df['mean_glucose'].plot(kind='bar', ax=ax3, color='orange')
    ax3.set_title('Mean Glucose by Subject')
    ax3.set_ylabel('Mean Glucose (mg/dL)')
    ax3.axhline(180, color='green', linestyle='--', alpha=0.7, label='Target')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # TIR Before vs After
    ax4 = axes[1, 0]
    x = np.arange(len(summary_df))
    width = 0.35
    ax4.bar(x - width/2, summary_df['baseline_tir'], width, label='Baseline TIR', alpha=0.7)
    ax4.bar(x + width/2, summary_df['control_period_tir'], width, label='Control Period TIR', alpha=0.7)
    ax4.set_xlabel('Subject')
    ax4.set_ylabel('TIR (%)')
    ax4.set_title('Time in Range: Baseline vs Control')
    ax4.set_xticks(x)
    ax4.set_xticklabels(summary_df['subject_id'])
    ax4.legend()
    
    # Hypoglycemia Time
    ax5 = axes[1, 1]
    summary_df['hypoglycemia_time'].plot(kind='bar', ax=ax5, color='red', alpha=0.7)
    ax5.set_title('Hypoglycemia Time by Subject')
    ax5.set_ylabel('Hypoglycemia Time (%)')
    ax5.tick_params(axis='x', rotation=45)
    
    # Overall Performance Scatter
    ax6 = axes[1, 2]
    scatter = ax6.scatter(summary_df['total_insulin'], summary_df['tir_improvement'], 
                         c=summary_df['hypoglycemia_time'], s=100, alpha=0.7, cmap='RdYlBu_r')
    ax6.set_xlabel('Total Insulin (units)')
    ax6.set_ylabel('TIR Improvement (%)')
    ax6.set_title('Performance vs Insulin Usage')
    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Hypoglycemia Time (%)')
    
    plt.tight_layout()
    return fig

def plot_glucose_distribution(results_dict, target_range=(70, 180)):
    """
    Plot glucose distribution before and during control for all subjects
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    all_baseline_glucose = []
    all_control_glucose = []
    
    for subject_id, results in results_dict.items():
        glucose_values = np.array(results['glucose_values'])
        timestamps = np.array(results['timestamps'])
        
        # Estimate control start (assuming 20% of data is baseline)
        control_start_idx = len(glucose_values) // 5
        
        baseline_glucose = glucose_values[:control_start_idx]
        control_glucose = glucose_values[control_start_idx:]
        
        all_baseline_glucose.extend(baseline_glucose)
        all_control_glucose.extend(control_glucose)
    
    # Histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(all_baseline_glucose, bins=30, alpha=0.7, label='Baseline', density=True, color='lightcoral')
    ax1.hist(all_control_glucose, bins=30, alpha=0.7, label='Control Period', density=True, color='lightblue')
    ax1.axvspan(target_range[0], target_range[1], alpha=0.3, color='green', label='Target Range')
    ax1.set_xlabel('Glucose (mg/dL)')
    ax1.set_ylabel('Density')
    ax1.set_title('Glucose Distribution: Baseline vs Control')
    ax1.legend()
    
    # Box plot comparison
    ax2 = axes[0, 1]
    data_to_plot = [all_baseline_glucose, all_control_glucose]
    bp = ax2.boxplot(data_to_plot, labels=['Baseline', 'Control'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax2.axhspan(target_range[0], target_range[1], alpha=0.3, color='green', label='Target Range')
    ax2.set_ylabel('Glucose (mg/dL)')
    ax2.set_title('Glucose Distribution Box Plot')
    ax2.legend()
    
    # Time in range comparison
    ax3 = axes[1, 0]
    baseline_tir = np.mean((np.array(all_baseline_glucose) >= target_range[0]) & 
                          (np.array(all_baseline_glucose) <= target_range[1])) * 180
    control_tir = np.mean((np.array(all_control_glucose) >= target_range[0]) & 
                         (np.array(all_control_glucose) <= target_range[1])) * 180
    
    categories = ['Hypoglycemia\n(<70)', 'Target Range\n(70-180)', 'Hyperglycemia\n(>180)']
    
    baseline_hypo = np.mean(np.array(all_baseline_glucose) < target_range[0]) * 100
    baseline_target = np.mean((np.array(all_baseline_glucose) >= target_range[0]) & 
                             (np.array(all_baseline_glucose) <= target_range[1])) * 100
    baseline_hyper = np.mean(np.array(all_baseline_glucose) > target_range[1]) * 100
    
    control_hypo = np.mean(np.array(all_control_glucose) < target_range[0]) * 100
    control_target = np.mean((np.array(all_control_glucose) >= target_range[0]) & 
                            (np.array(all_control_glucose) <= target_range[1])) * 100
    control_hyper = np.mean(np.array(all_control_glucose) > target_range[1]) * 100
    
    baseline_values = [baseline_hypo, baseline_target, baseline_hyper]
    control_values = [control_hypo, control_target, control_hyper]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, baseline_values, width, label='Baseline', 
                    color=['red', 'green', 'orange'], alpha=0.7)
    bars2 = ax3.bar(x + width/2, control_values, width, label='Control Period', 
                    color=['red', 'green', 'orange'], alpha=0.9)
    
    ax3.set_ylabel('Time (%)')
    ax3.set_title('Glycemic Control Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Glucose trends over time
    ax4 = axes[1, 1]
    
    # Calculate rolling averages for all subjects combined
    if len(results_dict) > 0:
        # Get the longest time series
        max_length = max(len(results['glucose_values']) for results in results_dict.values())
        
        # Create time-aligned glucose matrix
        glucose_matrix = []
        for subject_id, results in results_dict.items():
            glucose_values = np.array(results['glucose_values'])
            # Pad shorter series with NaN
            if len(glucose_values) < max_length:
                padded = np.full(max_length, np.nan)
                padded[:len(glucose_values)] = glucose_values
                glucose_matrix.append(padded)
            else:
                glucose_matrix.append(glucose_values)
        
        glucose_matrix = np.array(glucose_matrix)
        
        # Calculate mean and std at each time point
        mean_glucose = np.nanmean(glucose_matrix, axis=0)
        std_glucose = np.nanstd(glucose_matrix, axis=0)
        
        time_hours = np.arange(len(mean_glucose)) * 5 / 60  # Convert to hours
        
        ax4.plot(time_hours, mean_glucose, 'b-', linewidth=2, label='Mean Glucose')
        ax4.fill_between(time_hours, mean_glucose - std_glucose, mean_glucose + std_glucose, 
                        alpha=0.3, color='blue', label='±1 SD')
        ax4.axhline(target_range[0], color='green', linestyle=':', alpha=0.7)
        ax4.axhline(target_range[1], color='green', linestyle=':', alpha=0.7)
        ax4.fill_between(time_hours, target_range[0], target_range[1], alpha=0.2, color='green')
        
        # Mark control start
        control_start_time = len(mean_glucose) // 5 * 5 / 60  # 20% of data
        ax4.axvline(control_start_time, color='red', linestyle='--', alpha=0.7, 
                   label='Control Start')
        
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Glucose (mg/dL)')
        ax4.set_title('Average Glucose Trend Across All Subjects')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_performance_dashboard(results_dict, summary_df, target_range=(70, 180)):
    """
    Create a comprehensive dashboard showing overall performance
    """
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Overall TIR Improvement (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    if len(summary_df) > 0:
        improvement_colors = ['green' if x > 2 else 'orange' if x > 0 else 'red' 
                             for x in summary_df['tir_improvement']]
        bars = ax1.bar(range(len(summary_df)), summary_df['tir_improvement'], 
                      color=improvement_colors, alpha=0.7)
        ax1.set_xlabel('Subject ID')
        ax1.set_ylabel('TIR Improvement (%)')
        ax1.set_title('Time in Range Improvement by Subject')
        ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax1.axhline(5, color='green', linestyle='--', alpha=0.5, label='Good improvement (>5%)')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, summary_df['tir_improvement'])):
            ax1.annotate(f'{value:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3 if value >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
        
        ax1.set_xticks(range(len(summary_df)))
        ax1.set_xticklabels(summary_df['subject_id'])
        ax1.legend()
    
    # 2. Safety metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    if len(summary_df) > 0:
        ax2.scatter(summary_df['total_insulin'], summary_df['hypoglycemia_time'], 
                   s=100, alpha=0.7, c=summary_df['tir_improvement'], cmap='RdYlGn')
        ax2.set_xlabel('Total Insulin Usage (units)')
        ax2.set_ylabel('Hypoglycemia Time (%)')
        ax2.set_title('Safety Analysis: Insulin vs Hypoglycemia')
        
        # Add safety threshold
        ax2.axhline(5, color='red', linestyle='--', alpha=0.7, label='Safety threshold (5%)')
        ax2.legend()
        
        # Add colorbar
        scatter = ax2.scatter(summary_df['total_insulin'], summary_df['hypoglycemia_time'], 
                             s=100, alpha=0.7, c=summary_df['tir_improvement'], cmap='RdYlGn')
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('TIR Improvement (%)')
    
    # 3. Insulin action curve (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    controller = AutomatedInsulinController(target_range=target_range)
    time_points = np.linspace(0, controller.insulin_duration, 200)
    action_values = [controller.insulin_action_curve(t) for t in time_points]
    
    ax3.plot(time_points, action_values, 'b-', linewidth=2, label='Insulin Action')
    ax3.axvline(controller.insulin_peak, color='red', linestyle='--', alpha=0.7, 
               label=f'Peak at {controller.insulin_peak} min')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Insulin Action (fraction)')
    ax3.set_title('Insulin Action Curve Model')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance metrics summary (middle right)
    ax4 = fig.add_subplot(gs[1, 2:])
    if len(summary_df) > 0:
        # Create performance summary table
        metrics = {
            'Subjects Analyzed': len(summary_df),
            'Avg TIR Improvement': f"{summary_df['tir_improvement'].mean():.1f}%",
            'Subjects Improved': f"{(summary_df['tir_improvement'] > 2).sum()}/{len(summary_df)}",
            'Avg Insulin Usage': f"{summary_df['total_insulin'].mean():.1f} units",
            'Avg Hypoglycemia': f"{summary_df['hypoglycemia_time'].mean():.1f}%",
            'Best Performer': f"Subject {summary_df.loc[summary_df['tir_improvement'].idxmax(), 'subject_id']}"
        }
        
        # Create text summary
        y_pos = 0.9
        for key, value in metrics.items():
            ax4.text(0.1, y_pos, f"{key}:", fontweight='bold', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.6, y_pos, value, fontsize=12, transform=ax4.transAxes)
            y_pos -= 0.15
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Performance Summary', fontweight='bold', fontsize=14)
    
    # 5. Glucose distribution comparison (bottom left)
    ax5 = fig.add_subplot(gs[2:, :2])
    if results_dict:
        all_baseline_glucose = []
        all_control_glucose = []
        
        for subject_id, results in results_dict.items():
            glucose_values = np.array(results['glucose_values'])
            control_start_idx = len(glucose_values) // 5
            
            baseline_glucose = glucose_values[:control_start_idx]
            control_glucose = glucose_values[control_start_idx:]
            
            all_baseline_glucose.extend(baseline_glucose)
            all_control_glucose.extend(control_glucose)
        
        ax5.hist(all_baseline_glucose, bins=30, alpha=0.7, label='Baseline', 
                density=True, color='lightcoral')
        ax5.hist(all_control_glucose, bins=30, alpha=0.7, label='Control Period', 
                density=True, color='lightblue')
        ax5.axvspan(target_range[0], target_range[1], alpha=0.3, color='green', 
                   label='Target Range')
        ax5.set_xlabel('Glucose (mg/dL)')
        ax5.set_ylabel('Density')
        ax5.set_title('Glucose Distribution: Before vs During Control')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Individual subject performance heatmap (bottom right)
    ax6 = fig.add_subplot(gs[2:, 2:])
    if len(summary_df) > 0:
        # Create performance matrix
        performance_data = summary_df[['tir_improvement', 'hypoglycemia_time', 'total_insulin']].T
        
        # Normalize data for heatmap
        performance_data_norm = performance_data.copy()
        for i in range(len(performance_data_norm)):
            row = performance_data_norm.iloc[i]
            if row.std() > 0:
                performance_data_norm.iloc[i] = (row - row.mean()) / row.std()
        
        im = ax6.imshow(performance_data_norm.values, cmap='RdYlGn', aspect='auto')
        
        # Set labels
        ax6.set_xticks(range(len(summary_df)))
        ax6.set_xticklabels(summary_df['subject_id'])
        ax6.set_yticks(range(len(performance_data_norm)))
        ax6.set_yticklabels(['TIR Improvement', 'Hypoglycemia', 'Insulin Usage'])
        ax6.set_xlabel('Subject ID')
        ax6.set_title('Performance Heatmap (Normalized)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax6)
        cbar.set_label('Normalized Performance (σ)')
        
        # Add text annotations
        for i in range(len(performance_data_norm)):
            for j in range(len(summary_df)):
                original_value = performance_data.iloc[i, j]
                if i == 0:  # TIR improvement
                    text = f'{original_value:.1f}%'
                elif i == 1:  # Hypoglycemia
                    text = f'{original_value:.1f}%'
                else:  # Insulin usage
                    text = f'{original_value:.1f}U'
                
                ax6.text(j, i, text, ha='center', va='center', 
                        fontsize=8, fontweight='bold')
    
    plt.suptitle('Automated Insulin Controller - Performance Dashboard', 
                fontsize=16, fontweight='bold')
    
    return fig

# UPDATED: Enhanced analysis function with comprehensive visualizations

def run_controller_analysis(csv_file_path, target_range=(70, 180), create_plots=True):
    """
    Run the automated insulin controller analysis with comprehensive visualizations
    """
    # Load data
    df = pd.read_csv(csv_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['ID', 'timestamp']).reset_index(drop=True)
    subject_ids = df['ID'].unique()
    
    print(f"Loaded data for {len(subject_ids)} subjects")
    print(f"Running FIXED controller analysis with target range: {target_range[0]}-{target_range[1]} mg/dL")
    print("=" * 80)
    
    all_results = {}
    summary_stats = []
    
    for subject_id in subject_ids[:50]:  # Limit to first 5 subjects for testing
        print(f"\nProcessing Subject {subject_id}...")
        
        # Filter data for this subject
        subject_data = df[df['ID'] == subject_id].copy()
        subject_data = subject_data.sort_values('timestamp').reset_index(drop=True)
        
        if len(subject_data) < 200:  # Need more data for proper training/testing
            print(f"  Skipping - insufficient data ({len(subject_data)} points)")
            continue
            
        # Extract data
        cgm_data = pd.Series(subject_data['cgm'].values)
        food_data = pd.Series(subject_data['kcal'].values)
        
        # Clean data
        valid_indices = ~(cgm_data.isna() | food_data.isna())
        cgm_data = cgm_data[valid_indices]
        food_data = food_data[valid_indices]
        
        if len(cgm_data) < 200:
            print(f"  Skipping - insufficient clean data ({len(cgm_data)} points)")
            continue
        
        # Calculate baseline statistics
        baseline_tir = ((cgm_data >= target_range[0]) & (cgm_data <= target_range[1])).mean() * 100
        baseline_hypo = (cgm_data < target_range[0]).mean() * 100
        baseline_hyper = (cgm_data > target_range[1]).mean() * 100
        
        print(f"  Data points: {len(cgm_data)}")
        print(f"  Mean glucose: {cgm_data.mean():.1f} ± {cgm_data.std():.1f} mg/dL")
        print(f"  Baseline TIR: {baseline_tir:.1f}%")
        
        try:
            # Create and run controller
            controller = AutomatedInsulinController(target_range=target_range)
            results = controller.simulate_control(cgm_data, food_data, subject_id)
            
            # Display results
            print(f"  FIXED Controller Results:")
            print(f"    Overall TIR: {results['time_in_range']:.1f}%")
            if 'control_period_tir' in results:
                print(f"    Control Period TIR: {results['control_period_tir']:.1f}%")
                print(f"    TIR Improvement: {results.get('tir_improvement_control_period', 0):.1f}%")
            print(f"    Total insulin: {results['total_insulin']:.1f} units")
            print(f"    Hypoglycemia: {results['hypoglycemia_time']:.1f}%")
            
            # Store results
            all_results[subject_id] = results
            
            summary_stats.append({
                'subject_id': subject_id,
                'data_points': len(cgm_data),
                'mean_glucose': cgm_data.mean(),
                'glucose_std': cgm_data.std(),
                'baseline_tir': baseline_tir,
                'insulin_sensitivity': controller.insulin_sensitivity,
                'control_aggressiveness': controller.control_aggressiveness,
                'overall_tir': results['time_in_range'],
                'control_period_tir': results.get('control_period_tir', results['time_in_range']),
                'tir_improvement': results.get('tir_improvement_control_period', 0),
                'total_insulin': results['total_insulin'],
                'hypoglycemia_time': results['hypoglycemia_time']
            })
            
            # Create individual subject plot
            if create_plots:
                import os
                # Create aireadi subfolder if it doesn't exist
                os.makedirs('aireadi', exist_ok=True)
                """
                fig = plot_subject_overview(results, subject_id, target_range)
                plt.savefig(f'aireadi/subject_{subject_id}_overview.png', dpi=300, bbox_inches='tight')
                plt.close()
                """
                # ADDED: Create comparison plot
                fig_comp = plot_glucose_comparison(
                    results['original_glucose_values'], 
                    results['glucose_values'], 
                    results['insulin_doses'],
                    results['timestamps'],
                    subject_id, 
                    target_range,
                    results['control_start_time'],
                    results['in_range'],
                    results['time_in_range']
                )
                plt.savefig(f'aireadi/subject_{subject_id}_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats) if summary_stats else pd.DataFrame()
    
    # Save summary data to CSV
    if len(summary_df) > 0:
        import os
        os.makedirs('aireadi', exist_ok=True)
        summary_df.to_csv('aireadi/controller_results_summary.csv', index=False)
        print(f"Saved summary results to aireadi/controller_results_summary.csv")
    
    # Create comprehensive visualizations
    if create_plots and len(all_results) > 0:
        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        # 1. Summary statistics plots
        if len(summary_df) > 0:
            print("Creating summary statistics plots...")
            fig_summary = plot_summary_statistics(summary_df)
            if fig_summary:
                plt.savefig('aireadi/controller_summary_statistics.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Glucose distribution analysis
        print("Creating glucose distribution plots...")
        fig_dist = plot_glucose_distribution(all_results, target_range)
        if fig_dist:
            plt.savefig('aireadi/glucose_distribution_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Performance dashboard
        print("Creating performance dashboard...")
        fig_dashboard = create_performance_dashboard(all_results, summary_df, target_range)
        if fig_dashboard:
            plt.savefig('aireadi/performance_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Insulin action curve
        print("Creating insulin action curve...")
        controller = AutomatedInsulinController(target_range=target_range)
        fig_action = plot_insulin_action_curve(controller)
        plt.savefig('aireadi/insulin_action_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Summary
    if summary_stats:
        print(f"\n{'='*80}")
        print("FIXED CONTROLLER SUMMARY")
        print(f"{'='*80}")
        print(f"Processed subjects: {len(summary_df)}")
        print(f"Average TIR improvement: {summary_df['tir_improvement'].mean():.1f}% ± {summary_df['tir_improvement'].std():.1f}%")
        print(f"Average insulin usage: {summary_df['total_insulin'].mean():.1f} ± {summary_df['total_insulin'].std():.1f} units")
        print(f"Subjects with improvement: {(summary_df['tir_improvement'] > 2).sum()}/{len(summary_df)}")
        
        return all_results, summary_df
    else:
        return {}, pd.DataFrame()

# Example usage with comprehensive visualizations
if __name__ == "__main__":
    csv_file = "aireadi_cgm_kcal.csv"
    
    try:
        print("Starting Automated Insulin Controller Analysis with Visualizations...")
        print("This will create multiple plots showing:")
        print("  - Individual subject glucose traces and insulin delivery")
        print("  - Summary statistics across all subjects")
        print("  - Glucose distribution analysis")
        print("  - Performance dashboard")
        print("  - Insulin action curve model")
        print()
        
        all_results, summary_df = run_controller_analysis(csv_file, 
                                                         target_range=(70, 180),
                                                         create_plots=True)
        
        if len(summary_df) > 0:
            print(f"\n{'='*80}")
            print("ANALYSIS COMPLETE!")
            print(f"{'='*80}")
            print("Generated visualizations:")
            print("  - subject_X_overview.png: Individual subject analysis")
            print("  - controller_summary_statistics.png: Summary across subjects")
            print("  - glucose_distribution_analysis.png: Glucose distribution comparison")
            print("  - performance_dashboard.png: Comprehensive performance dashboard")
            print("  - insulin_action_curve.png: Model insulin action curve")
            print()
            
            best_subject = summary_df.loc[summary_df['tir_improvement'].idxmax()]
            print(f"Best performing subject: {best_subject['subject_id']} "
                  f"({best_subject['tir_improvement']:.1f}% TIR improvement)")
            
        else:
            print("No data could be processed. Please check your CSV file.")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()