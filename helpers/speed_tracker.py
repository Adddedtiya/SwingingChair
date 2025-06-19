from time import time as current_time


class TimeTracker:
    start_time   : float = 0.0
    elapsed_time : float = 0.0

    @staticmethod
    def start_clock() -> None:
        TimeTracker.start_time = current_time()
    
    @staticmethod
    def stop_clock() -> None:
        current = current_time()
        TimeTracker.elapsed_time = current - TimeTracker.start_time

    @staticmethod
    def estimate_time(ammount : int) -> str:
        estimated_time = TimeTracker.elapsed_time * ammount

        # Convert to days, hours, minutes, seconds
        days,  rem = divmod(estimated_time, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)

        return f"Estimated remaning time : {int(days)}d {int(hours)}h {int(minutes)}m"