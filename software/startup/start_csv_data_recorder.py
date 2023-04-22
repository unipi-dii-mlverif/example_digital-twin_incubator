import time

from incubator.config.config import config_logger, load_config
from digital_twin.data_access.dbmanager.incubator_state_csv_recorder import IncubatorDataRecorderCSV

if __name__ == '__main__':
    config_logger("logging.conf")
    config = load_config("startup.conf")
    while True:
        try:
            recorder = IncubatorDataRecorderCSV(csv_file_path=".", csv_file_prefix="rec", rollover_limit=100000)
            recorder.start_recording(rabbit_config=config["rabbitmq"])
        except KeyboardInterrupt:
            exit(0)
        except Exception as exc:
            print("Error: ")
            print(exc)
            print("Attempting to reconnect...")
            time.sleep(1.0)