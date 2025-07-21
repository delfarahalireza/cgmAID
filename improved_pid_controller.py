import numpy as np

class ImprovedPIDController:
    """
    Improved PID controller with realistic constraints and safety features
    for automated insulin delivery.
    """
    def __init__(self, target=140, kp=0.02, ki=0.0005, kd=0.01, basal=0.8, dt=5/60,
                 max_basal_rate=3.0, max_bolus=10.0, suspend_threshold=70):
        """
        target: target glucose (mg/dL) - increased to realistic 140
        kp, ki, kd: PID gains - increased for more realistic response
        basal: basal insulin rate (units/hour) - more realistic basal
        dt: time step (hours)
        max_basal_rate: maximum basal rate (units/hour)
        max_bolus: maximum bolus (units)
        suspend_threshold: glucose level to suspend insulin (mg/dL)
        """
        self.target = target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.basal = basal
        self.dt = dt
        self.max_basal_rate = max_basal_rate
        self.max_bolus = max_bolus
        self.suspend_threshold = suspend_threshold
        
        # PID state variables
        self.integral = 0
        self.previous_error = 0
        self.previous_glucose = None
        
        # Rate limiting
        self.previous_output = basal
        self.max_rate_change = 2.0  # max change in units/hour per time step

    def reset(self):
        """Reset PID controller state"""
        self.integral = 0
        self.previous_error = 0
        self.previous_glucose = None
        self.previous_output = self.basal

    def compute(self, glucose, iob=0):
        """
        Compute insulin delivery rate
        glucose: current glucose (mg/dL)
        iob: insulin on board (units)
        Returns: insulin rate (units/hour)
        """
        # Safety check - suspend if glucose too low
        if glucose < self.suspend_threshold:
            self.integral = 0  # Reset integral when suspended
            return 0.0
        
        # Calculate error
        error = glucose - self.target
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * self.dt
        # Prevent integral windup
        max_integral = 50.0 / self.ki if self.ki > 0 else 0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        integral = self.ki * self.integral
        
        # Derivative term (use glucose rate of change)
        derivative = 0
        if self.previous_glucose is not None:
            glucose_rate = (glucose - self.previous_glucose) / self.dt
            derivative = self.kd * glucose_rate
        
        # Calculate raw PID output
        pid_output = proportional + integral - derivative  # Negative derivative for predictive action
        
        # Add basal insulin
        insulin_rate = self.basal + pid_output
        
        # IOB adjustment - reduce insulin if too much IOB
        if iob > 1.5:  # If more than 1.5 units on board
            iob_reduction = min(0.5 * (iob - 1.5), insulin_rate * 0.5)
            insulin_rate -= iob_reduction
        
        # Apply constraints
        insulin_rate = np.clip(insulin_rate, 0, self.max_basal_rate)
        
        # Rate limiting - prevent too rapid changes
        rate_change = insulin_rate - self.previous_output
        max_change = self.max_rate_change * self.dt
        if abs(rate_change) > max_change:
            insulin_rate = self.previous_output + np.sign(rate_change) * max_change
        
        # Update state
        self.previous_error = error
        self.previous_glucose = glucose
        self.previous_output = insulin_rate
        
        return insulin_rate

    def compute_meal_bolus(self, carbs_g, current_glucose, insulin_to_carb_ratio=10):
        """
        Compute meal bolus using carbohydrate ratio and correction factor
        carbs_g: carbohydrates in grams
        current_glucose: current glucose level
        insulin_to_carb_ratio: grams of carb per unit of insulin
        Returns: bolus amount (units)
        """
        if carbs_g <= 0:
            return 0.0
        
        # Carbohydrate bolus
        carb_bolus = carbs_g / insulin_to_carb_ratio
        
        # Correction bolus (if glucose > target)
        correction_factor = 30  # mg/dL per unit of insulin
        if current_glucose > self.target + 20:  # Only correct if significantly above target
            correction_bolus = (current_glucose - self.target) / correction_factor
        else:
            correction_bolus = 0
        
        total_bolus = carb_bolus + correction_bolus
        
        # Apply safety limits
        total_bolus = min(total_bolus, self.max_bolus)
        
        return total_bolus

if __name__ == '__main__':
    # Test the improved PID controller
    print("Testing Improved PID Controller")
    print("=" * 40)
    
    controller = ImprovedPIDController()
    
    # Test 1: Response to different glucose levels
    print("\n1. Glucose Response Test:")
    print("   Glucose | Insulin Rate | Action")
    print("   --------|--------------|--------")
    
    test_glucose_levels = [60, 80, 100, 140, 180, 220, 300]
    controller.reset()
    
    for glucose in test_glucose_levels:
        insulin_rate = controller.compute(glucose, iob=0)
        if glucose < 70:
            action = "SUSPEND"
        elif insulin_rate > controller.basal:
            action = "INCREASE"
        elif insulin_rate < controller.basal:
            action = "DECREASE"
        else:
            action = "MAINTAIN"
        
        print(f"   {glucose:3d}     | {insulin_rate:8.2f}     | {action}")
    
    # Test 2: IOB effect
    print("\n2. Insulin on Board Effect:")
    print("   IOB   | Base Rate | Adjusted Rate")
    print("   ------|-----------|---------------")
    
    controller.reset()
    base_glucose = 200  # High glucose requiring insulin
    
    for iob in [0, 1, 2, 3, 4]:
        base_rate = controller.compute(base_glucose, iob=0)
        controller.reset()  # Reset for consistent comparison
        adjusted_rate = controller.compute(base_glucose, iob=iob)
        controller.reset()
        print(f"   {iob:3.1f}   | {base_rate:7.2f}   | {adjusted_rate:9.2f}")
    
    # Test 3: Meal bolus calculation
    print("\n3. Meal Bolus Calculation:")
    print("   Carbs | Glucose | Bolus | Components")
    print("   ------|---------|-------|------------")
    
    test_meals = [(30, 120), (50, 140), (75, 180), (60, 220)]
    
    for carbs, glucose in test_meals:
        bolus = controller.compute_meal_bolus(carbs, glucose)
        carb_component = carbs / 10  # Using default 1:10 ratio
        correction = max(0, (glucose - 140) / 30)
        print(f"   {carbs:2d}g  | {glucose:5d}   | {bolus:5.2f} | {carb_component:.2f} + {correction:.2f}")
    
    # Test 4: Rate limiting
    print("\n4. Rate Limiting Test:")
    print("   Step | Target Rate | Actual Rate | Limited?")
    print("   -----|-------------|-------------|----------")
    
    controller.reset()
    controller.previous_output = 1.0  # Start at 1 U/hr
    
    target_rates = [1.0, 3.5, 0.2, 2.8, 0.0]  # Rapid changes
    
    for i, target in enumerate(target_rates):
        # Simulate by temporarily setting what the unlimited rate would be
        glucose = 140 + (target - controller.basal) * 50  # Approximate glucose for this rate
        actual_rate = controller.compute(glucose, iob=0)
        limited = "YES" if abs(actual_rate - controller.previous_output) > controller.max_rate_change * controller.dt else "NO"
        
        print(f"   {i+1:2d}   | {target:7.2f}     | {actual_rate:7.2f}     | {limited:6s}")
    
    print(f"\n5. Controller Parameters:")
    print(f"   Target glucose: {controller.target} mg/dL")
    print(f"   PID gains: Kp={controller.kp}, Ki={controller.ki}, Kd={controller.kd}")
    print(f"   Basal rate: {controller.basal} U/hr")
    print(f"   Max rate: {controller.max_basal_rate} U/hr")
    print(f"   Suspend threshold: {controller.suspend_threshold} mg/dL")