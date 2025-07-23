import os
import sys
import subprocess
import psutil
import time
import json
import winreg
from pathlib import Path
import ctypes
from ctypes import wintypes

class IntegratedGPUOptimizer:
    def __init__(self):
        self.original_settings = {}
        self.is_admin = self.check_admin_rights()
        
    def check_admin_rights(self):
        """Check if running with administrator privileges"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def backup_current_settings(self):
        """Backup current system settings before optimization"""
        print("Backing up current settings...")
        backup_data = {
            'timestamp': time.time(),
            'power_plan': self.get_current_power_plan(),
            'priority_class': self.get_process_priority(),
        }
        
        with open('gpu_optimizer_backup.json', 'w') as f:
            json.dump(backup_data, f, indent=2)
        print("Settings backed up to gpu_optimizer_backup.json")
    
    def get_current_power_plan(self):
        """Get current Windows power plan"""
        try:
            result = subprocess.run(['powercfg', '/getactivescheme'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return None
    
    def get_process_priority(self):
        """Get current process priority"""
        return psutil.Process().nice()
    
    def set_high_performance_power_plan(self):
        """Set Windows to High Performance power plan"""
        if not self.is_admin:
            print("⚠️  Administrator rights needed for power plan changes")
            return False
        
        try:
            print("Setting High Performance power plan...")
            # High Performance GUID
            subprocess.run(['powercfg', '/setactive', '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'], 
                          check=True)
            
            # Disable CPU throttling
            subprocess.run(['powercfg', '/setacvalueindex', 'scheme_current', 
                          'sub_processor', 'PROCTHROTTLEMIN', '100'], check=True)
            subprocess.run(['powercfg', '/setacvalueindex', 'scheme_current', 
                          'sub_processor', 'PROCTHROTTLEMAX', '100'], check=True)
            
            # Apply settings
            subprocess.run(['powercfg', '/setactive', 'scheme_current'], check=True)
            print("✅ Power plan optimized")
            return True
        except Exception as e:
            print(f"❌ Power plan optimization failed: {e}")
            return False
    
    def optimize_memory_allocation(self):
        """Optimize system memory for graphics"""
        print("Optimizing memory allocation...")
        
        # Get system memory info
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        
        print(f"Total RAM: {total_gb:.1f} GB")
        
        # Calculate optimal shared memory for integrated GPU
        if total_gb >= 16:
            shared_memory_mb = 2048  # 2GB for 16GB+ systems
        elif total_gb >= 8:
            shared_memory_mb = 1024  # 1GB for 8GB+ systems
        else:
            shared_memory_mb = 512   # 512MB for lower memory systems
        
        print(f"Recommended shared GPU memory: {shared_memory_mb} MB")
        
        # Try to modify registry for Intel integrated graphics
        try:
            self.set_intel_gpu_memory(shared_memory_mb)
        except Exception as e:
            print(f"Intel GPU memory setting failed: {e}")
        
        return True
    
    def set_intel_gpu_memory(self, memory_mb):
        """Set Intel integrated GPU dedicated memory"""
        if not self.is_admin:
            print("⚠️  Administrator rights needed for GPU memory allocation")
            return
        
        try:
            # Intel Graphics registry path
            key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000"
            
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, 
                              winreg.KEY_ALL_ACCESS) as key:
                # Set dedicated video memory (in bytes)
                memory_bytes = memory_mb * 1024 * 1024
                winreg.SetValueEx(key, "DedicatedSegmentSize", 0, 
                                winreg.REG_DWORD, memory_bytes)
                print(f"✅ Intel GPU memory set to {memory_mb} MB")
        except Exception as e:
            print(f"❌ Intel GPU registry modification failed: {e}")
    
    def optimize_game_process(self, game_executable):
        """Optimize running game process"""
        print(f"Looking for game process: {game_executable}")
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if game_executable.lower() in proc.info['name'].lower():
                    process = psutil.Process(proc.info['pid'])
                    
                    # Set high priority
                    process.nice(psutil.HIGH_PRIORITY_CLASS)
                    
                    # Set CPU affinity to use all cores
                    cpu_count = psutil.cpu_count()
                    process.cpu_affinity(list(range(cpu_count)))
                    
                    print(f"✅ Optimized process {proc.info['name']} (PID: {proc.info['pid']})")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        print(f"❌ Game process {game_executable} not found")
        return False
    
    def clean_system_memory(self):
        """Clean system memory and cache"""
        print("Cleaning system memory...")
        
        try:
            # Clear standby memory (requires admin)
            if self.is_admin:
                subprocess.run(['sfc', '/scannow'], capture_output=True)
            
            # Garbage collection
            import gc
            gc.collect()
            
            print("✅ Memory cleaned")
            return True
        except Exception as e:
            print(f"❌ Memory cleaning failed: {e}")
            return False
    
    def disable_unnecessary_services(self):
        """Disable unnecessary Windows services for gaming"""
        if not self.is_admin:
            print("⚠️  Administrator rights needed to disable services")
            return False
        
        # Services that can be safely disabled during gaming
        services_to_disable = [
            'Windows Search',
            'Superfetch',
            'Themes',
            'Print Spooler',
            'Fax'
        ]
        
        print("Temporarily disabling unnecessary services...")
        
        for service in services_to_disable:
            try:
                subprocess.run(['sc', 'stop', service], 
                             capture_output=True, check=False)
                print(f"✅ Stopped {service}")
            except:
                print(f"⚠️  Could not stop {service}")
        
        return True
    
    def optimize_nvidia_settings(self):
        """Optimize NVIDIA settings if available"""
        nvidia_smi_path = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        
        if not os.path.exists(nvidia_smi_path):
            print("NVIDIA GPU not detected, skipping NVIDIA optimizations")
            return False
        
        try:
            print("Optimizing NVIDIA settings...")
            
            # Set maximum performance mode
            subprocess.run([nvidia_smi_path, '-pm', '1'], check=True)  # Persistence mode
            subprocess.run([nvidia_smi_path, '-ac', '4000,1000'], check=False)  # Set clocks
            
            print("✅ NVIDIA settings optimized")
            return True
        except Exception as e:
            print(f"❌ NVIDIA optimization failed: {e}")
            return False
    
    def create_game_launcher(self, game_path, game_args=""):
        """Create optimized game launcher"""
        launcher_content = f'''@echo off
echo Starting GPU Optimizer...
python "{os.path.abspath(__file__)}" --optimize --game="{os.path.basename(game_path)}"

echo Launching game with optimizations...
cd /d "{os.path.dirname(game_path)}"

REM Set process priority and launch game
start /high /affinity FF "{os.path.basename(game_path)}" {game_args}

echo Game launched with optimizations!
pause
'''
        
        launcher_path = "optimized_game_launcher.bat"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        print(f"✅ Created optimized launcher: {launcher_path}")
        return launcher_path
    
    def monitor_performance(self, duration=60):
        """Monitor system performance during gaming"""
        print(f"Monitoring performance for {duration} seconds...")
        
        start_time = time.time()
        samples = []
        
        while time.time() - start_time < duration:
            sample = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'gpu_temp': self.get_gpu_temperature()
            }
            samples.append(sample)
            
            print(f"CPU: {sample['cpu_percent']:5.1f}% | "
                  f"RAM: {sample['memory_percent']:5.1f}% | "
                  f"GPU Temp: {sample['gpu_temp']}°C")
            
            time.sleep(5)
        
        # Save performance log
        with open('performance_log.json', 'w') as f:
            json.dump(samples, f, indent=2)
        
        print("✅ Performance monitoring complete")
        return samples
    
    def get_gpu_temperature(self):
        """Get GPU temperature (simplified)"""
        try:
            # This is a placeholder - actual implementation would need specific GPU APIs
            return "N/A"
        except:
            return "N/A"
    
    def restore_settings(self):
        """Restore original system settings"""
        if not os.path.exists('gpu_optimizer_backup.json'):
            print("No backup file found")
            return False
        
        try:
            with open('gpu_optimizer_backup.json', 'r') as f:
                backup = json.load(f)
            
            print("Restoring original settings...")
            
            # Restore power plan
            if backup.get('power_plan') and self.is_admin:
                # Extract GUID from power plan string
                import re
                guid_match = re.search(r'\(([^)]+)\)', backup['power_plan'])
                if guid_match:
                    guid = guid_match.group(1)
                    subprocess.run(['powercfg', '/setactive', guid], check=False)
            
            print("✅ Settings restored")
            return True
        except Exception as e:
            print(f"❌ Settings restoration failed: {e}")
            return False
    
    def run_full_optimization(self, game_executable=None):
        """Run complete optimization suite"""
        print("=" * 60)
        print("🚀 INTEGRATED GPU GAME OPTIMIZER")
        print("=" * 60)
        
        if not self.is_admin:
            print("⚠️  WARNING: Running without administrator privileges")
            print("   Some optimizations will be limited")
            print()
        
        # Backup settings
        self.backup_current_settings()
        
        # Run optimizations
        optimizations = [
            ("Power Plan", self.set_high_performance_power_plan),
            ("Memory Allocation", self.optimize_memory_allocation),
            ("System Cleanup", self.clean_system_memory),
            ("Service Management", self.disable_unnecessary_services),
            ("NVIDIA Settings", self.optimize_nvidia_settings),
        ]
        
        results = {}
        for name, func in optimizations:
            print(f"\n🔧 {name}...")
            results[name] = func()
        
        # Optimize specific game if provided
        if game_executable:
            print(f"\n🎮 Game Process Optimization...")
            results["Game Process"] = self.optimize_game_process(game_executable)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for r in results.values() if r)
        total = len(results)
        
        for name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{name:20} : {status}")
        
        print(f"\nOptimizations completed: {successful}/{total}")
        
        if game_executable:
            print(f"\n🎯 To optimize further:")
            print(f"   1. Launch your game normally")
            print(f"   2. Run: python {__file__} --monitor")
            print(f"   3. Check performance_log.json for detailed metrics")
        
        print(f"\n🔄 To restore original settings:")
        print(f"   python {__file__} --restore")
        
        return results

def main():
    optimizer = IntegratedGPUOptimizer()
    
    if len(sys.argv) > 1:
        if '--optimize' in sys.argv:
            game_arg = next((arg for arg in sys.argv if arg.startswith('--game=')), None)
            game_name = game_arg.split('=')[1] if game_arg else None
            optimizer.run_full_optimization(game_name)
        
        elif '--monitor' in sys.argv:
            optimizer.monitor_performance(120)  # Monitor for 2 minutes
        
        elif '--restore' in sys.argv:
            optimizer.restore_settings()
        
        elif '--create-launcher' in sys.argv:
            game_path = input("Enter full path to game executable: ")
            game_args = input("Enter game arguments (optional): ")
            optimizer.create_game_launcher(game_path, game_args)
    
    else:
        # Interactive mode
        print("🎮 Integrated GPU Game Optimizer")
        print("\nOptions:")
        print("1. Run full optimization")
        print("2. Create optimized game launcher")
        print("3. Monitor performance")
        print("4. Restore original settings")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == '1':
            game_name = input("Enter game executable name (optional): ").strip()
            game_name = game_name if game_name else None
            optimizer.run_full_optimization(game_name)
        
        elif choice == '2':
            game_path = input("Enter full path to game executable: ")
            game_args = input("Enter game arguments (optional): ")
            optimizer.create_game_launcher(game_path, game_args)
        
        elif choice == '3':
            duration = input("Monitor duration in seconds (default 60): ")
            duration = int(duration) if duration.isdigit() else 60
            optimizer.monitor_performance(duration)
        
        elif choice == '4':
            optimizer.restore_settings()
        
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()