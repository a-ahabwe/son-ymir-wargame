"""
Power analysis module for experiment design.
Helps determine sample size requirements for robust statistical testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class PowerAnalysis:
    """
    Performs power analysis to determine appropriate sample sizes for experiments.
    """
    
    def __init__(self, alpha=0.05, target_power=0.8):
        """
        Initialize power analysis with standard parameters.
        
        Args:
            alpha (float): Significance level (Type I error rate)
            target_power (float): Desired statistical power (1 - Type II error rate)
        """
        self.alpha = alpha
        self.target_power = target_power
        
    def calculate_sample_size(self, effect_size, test_type="t_test", groups=2, paired=False):
        """
        Calculate required sample size for a given effect size and test type.
        
        Args:
            effect_size (float): Expected effect size (Cohen's d for t-tests)
            test_type (str): Type of statistical test ("t_test", "anova", "chi_square")
            groups (int): Number of groups (for ANOVA)
            paired (bool): Whether the test is paired or not (for t-tests)
            
        Returns:
            int: Required sample size
        """
        if test_type == "t_test":
            return self._t_test_sample_size(effect_size, paired)
        elif test_type == "anova":
            return self._anova_sample_size(effect_size, groups)
        elif test_type == "chi_square":
            return self._chi_square_sample_size(effect_size)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
            
    def calculate_power(self, effect_size, sample_size, test_type="t_test", groups=2, paired=False):
        """
        Calculate statistical power for a given effect size, sample size, and test type.
        
        Args:
            effect_size (float): Expected effect size
            sample_size (int): Sample size per group
            test_type (str): Type of statistical test
            groups (int): Number of groups (for ANOVA)
            paired (bool): Whether the test is paired or not (for t-tests)
            
        Returns:
            float: Statistical power
        """
        if test_type == "t_test":
            return self._t_test_power(effect_size, sample_size, paired)
        elif test_type == "anova":
            return self._anova_power(effect_size, sample_size, groups)
        elif test_type == "chi_square":
            return self._chi_square_power(effect_size, sample_size)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
            
    def _t_test_sample_size(self, effect_size, paired=False):
        """Calculate sample size for t-test"""
        # For paired t-test, the required sample size is smaller
        if paired:
            # Paired t-test formula
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            z_beta = stats.norm.ppf(self.target_power)
            n = ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(n))
        else:
            # Independent t-test formula
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            z_beta = stats.norm.ppf(self.target_power)
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(n))
            
    def _t_test_power(self, effect_size, sample_size, paired=False):
        """Calculate power for t-test"""
        if paired:
            # Paired t-test
            df = sample_size - 1
            nc = effect_size * np.sqrt(sample_size)
        else:
            # Independent t-test
            df = 2 * sample_size - 2
            nc = effect_size * np.sqrt(sample_size / 2)
            
        # Critical value for rejection
        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        
        # Calculate power
        power = 1 - stats.nct.cdf(t_crit, df, nc)
        return power
        
    def _anova_sample_size(self, effect_size, groups):
        """Calculate sample size for ANOVA"""
        # ANOVA sample size formula
        f_squared = effect_size ** 2
        numerator_df = groups - 1
        
        # Approximate using F-distribution
        target_power = self.target_power
        alpha = self.alpha
        
        # Initial guess
        n = 20
        
        # Iteratively find the required sample size
        while True:
            denominator_df = groups * (n - 1)
            f_critical = stats.f.ppf(1 - alpha, numerator_df, denominator_df)
            
            # Calculate non-centrality parameter
            lambda_nc = n * groups * f_squared
            
            # Calculate power
            power = 1 - stats.ncf.cdf(
                f_critical, numerator_df, denominator_df, lambda_nc
            )
            
            if power >= target_power:
                return int(np.ceil(n * groups))
            
            n += 5
            
            # Safety check to prevent infinite loop
            if n > 1000:
                return int(np.ceil(n * groups))
                
    def _anova_power(self, effect_size, sample_size, groups):
        """Calculate power for ANOVA"""
        # Sample size per group
        n_per_group = sample_size // groups
        
        # Degrees of freedom
        numerator_df = groups - 1
        denominator_df = groups * (n_per_group - 1)
        
        # Non-centrality parameter
        f_squared = effect_size ** 2
        lambda_nc = n_per_group * groups * f_squared
        
        # Critical F-value
        f_critical = stats.f.ppf(1 - self.alpha, numerator_df, denominator_df)
        
        # Calculate power
        power = 1 - stats.ncf.cdf(
            f_critical, numerator_df, denominator_df, lambda_nc
        )
        
        return power
        
    def _chi_square_sample_size(self, effect_size):
        """Calculate sample size for chi-square test"""
        # Chi-square sample size formula
        # We'll use a simple approximation for a 2x2 contingency table
        w = effect_size  # w is the effect size for chi-square
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.target_power)
        
        # Sample size formula for chi-square
        n = ((z_alpha + z_beta) / w) ** 2
        
        return int(np.ceil(n))
        
    def _chi_square_power(self, effect_size, sample_size):
        """Calculate power for chi-square test"""
        # Calculate non-centrality parameter
        lambda_nc = sample_size * effect_size ** 2
        
        # Degrees of freedom (assuming 2x2 contingency table)
        df = 1
        
        # Critical chi-square value
        chi2_critical = stats.chi2.ppf(1 - self.alpha, df)
        
        # Calculate power
        power = 1 - stats.ncx2.cdf(chi2_critical, df, lambda_nc)
        
        return power
        
    def plot_power_curve(self, effect_sizes, sample_sizes, test_type="t_test", 
                        groups=2, paired=False, output_path=None):
        """
        Plot power curves for different effect sizes and sample sizes.
        
        Args:
            effect_sizes (list): List of effect sizes to plot
            sample_sizes (list): List of sample sizes to plot
            test_type (str): Type of statistical test
            groups (int): Number of groups (for ANOVA)
            paired (bool): Whether the test is paired (for t-tests)
            output_path (str): Path to save the plot (if None, plot is displayed)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        plt.figure(figsize=(10, 6))
        
        for effect_size in effect_sizes:
            powers = [self.calculate_power(effect_size, n, test_type, groups, paired) 
                     for n in sample_sizes]
            plt.plot(sample_sizes, powers, label=f"Effect size = {effect_size}")
            
        # Add target power line
        plt.axhline(y=self.target_power, color='r', linestyle='--', 
                   label=f"Target power = {self.target_power}")
        
        plt.xlabel("Sample Size")
        plt.ylabel("Statistical Power")
        plt.title(f"Power Analysis for {test_type.replace('_', ' ').title()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
            
        return plt.gcf()
        
    def generate_report(self, effect_size, test_type="t_test", groups=2, paired=False,
                       sample_sizes=None, output_path=None):
        """
        Generate a comprehensive power analysis report.
        
        Args:
            effect_size (float): Expected effect size
            test_type (str): Type of statistical test
            groups (int): Number of groups (for ANOVA)
            paired (bool): Whether the test is paired (for t-tests)
            sample_sizes (list): List of sample sizes to analyze (if None, automatically generated)
            output_path (str): Path to save the report (if None, returned as a string)
            
        Returns:
            str or None: Report text if output_path is None, otherwise None
        """
        if sample_sizes is None:
            # Generate reasonable sample sizes
            required_n = self.calculate_sample_size(effect_size, test_type, groups, paired)
            min_n = max(5, required_n // 2)
            max_n = required_n * 2
            sample_sizes = list(range(min_n, max_n + 1, max(1, (max_n - min_n) // 20)))
            
        # Calculate required sample size
        required_n = self.calculate_sample_size(effect_size, test_type, groups, paired)
        
        # Calculate power for different sample sizes
        powers = [self.calculate_power(effect_size, n, test_type, groups, paired) 
                 for n in sample_sizes]
        
        # Generate report text
        report = []
        report.append("# Power Analysis Report")
        report.append(f"\n## Test Configuration")
        report.append(f"- Test type: {test_type.replace('_', ' ').title()}")
        report.append(f"- Significance level (α): {self.alpha}")
        report.append(f"- Target power (1-β): {self.target_power}")
        report.append(f"- Expected effect size: {effect_size}")
        
        if test_type == "t_test":
            report.append(f"- Test design: {'Paired' if paired else 'Independent'}")
        elif test_type == "anova":
            report.append(f"- Number of groups: {groups}")
            
        report.append(f"\n## Results")
        report.append(f"- Required sample size: {required_n}")
        report.append(f"  - This is the minimum sample size needed to detect an effect of size {effect_size}")
        report.append(f"    with {self.target_power * 100:.0f}% power at an α level of {self.alpha}.")
        
        report.append(f"\n## Sample Size vs. Power")
        report.append("| Sample Size | Power |")
        report.append("| ----------- | ----- |")
        
        for n, power in zip(sample_sizes, powers):
            report.append(f"| {n} | {power:.3f} |")
            
        # Generate and save plot
        if output_path:
            # Save text report
            with open(f"{output_path}.md", "w") as f:
                f.write("\n".join(report))
                
            # Save plot
            self.plot_power_curve([effect_size], sample_sizes, test_type, groups, paired, 
                                f"{output_path}.png")
            
            return None
        else:
            return "\n".join(report)

# Example usage:
if __name__ == "__main__":
    power_analyzer = PowerAnalysis()
    
    # Calculate sample size for a medium effect
    sample_size = power_analyzer.calculate_sample_size(effect_size=0.5)
    print(f"Required sample size for medium effect: {sample_size}")
    
    # Generate a report
    report = power_analyzer.generate_report(effect_size=0.5)
    print(report)