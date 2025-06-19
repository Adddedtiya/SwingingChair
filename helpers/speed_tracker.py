from time import time as current_time


class TimeTracker:
    start_time   : float = 0.0
    elapsed_time : float = 0.0

    def start_clock(cls) -> None:
        cls.start_time = current_time()
    
    def stop_clock(cls) -> None:
        current = current_time()
        cls.elapsed_time = current - cls.start_time

    def estimate_time(cls, ammount : int) -> str:
        estimated_time = cls.elapsed_time * ammount

        # Convert to days, hours, minutes, seconds
        days,  rem = divmod(estimated_time, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)

        return f"Estimated remaning time : {int(days)}d {int(hours)}h {int(minutes)}m"