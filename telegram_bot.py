import requests
import time
import json
import yaml
import os
import pandas as pd
from pathlib import Path
from etl import ETL
import logging
from typing import Dict, Any, List

# Setup logging to identify where errors occur
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TG_Bot")

class TelegramBot:
    def __init__(self, config_path: str = "config.yml"):
        logger.info(f"Initializing TelegramBot with config: {config_path}")
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            tg_config = self.config.get('telegram', {})
            self.token = tg_config.get('token')
            self.api_url = f"https://api.telegram.org/bot{self.token}/"
            self.media_dir = Path(tg_config.get('media_dir', 'telegram_media'))
            self.media_dir.mkdir(exist_ok=True)
            self.table_name = tg_config.get('table_name', 'TG_MESSAGES')
            self.interval = tg_config.get('polling_interval', 5)
            
            # Oracle connection
            oracle_uri = self.config['server']['oracle']
            self.etl = ETL(oracle_uri, add_missing_cols=True, logging_enabled=True)
            logger.info("Connected to Oracle via ETL engine.")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}", exc_info=True)
            raise

    def get_updates(self, offset: int = None):
        params = {'timeout': 30, 'offset': offset}
        try:
            resp = requests.get(self.api_url + "getUpdates", params=params)
            resp.raise_for_status()
            return resp.json().get('result', [])
        except Exception as e:
            logger.error(f"Error fetching updates: {e}")
            return []

    def download_file(self, file_id: str, sub_dir: str) -> str:
        """Downloads a file from Telegram and returns the local path."""
        try:
            # Get file path from Telegram
            resp = requests.get(self.api_url + "getFile", params={'file_id': file_id})
            resp.raise_for_status()
            file_path = resp.json()['result']['file_path']
            
            # Download file
            download_url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
            local_path = self.media_dir / sub_dir / Path(file_path).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Downloaded file {file_id} to {local_path}")
            return str(local_path)
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}", exc_info=True)
            return f"ERROR: {e}"

    def flatten_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Flattens a complex Telegram message into a single-level dictionary."""
        flat = {}
        
        def _flatten(obj, prefix=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(v, f"{prefix}{k}_" if prefix else f"{k}_")
            elif isinstance(obj, list):
                # For lists (like entities or photo sizes), we just store as JSON string or handle specially
                if prefix.endswith('photo_'):
                    # Pick the largest photo (last in list)
                    _flatten(obj[-1], prefix)
                else:
                    flat[prefix[:-1]] = json.dumps(obj)
            else:
                # Truncate prefix trailing underscore
                key = prefix[:-1] if prefix else 'data'
                flat[key] = obj

        _flatten(msg)
        
        # Handle Media specifically
        if 'document' in msg:
            flat['media_local_path'] = self.download_file(msg['document']['file_id'], 'docs')
            flat['media_type'] = 'document'
        elif 'photo' in msg:
            # photo is a list
            flat['media_local_path'] = self.download_file(msg['photo'][-1]['file_id'], 'photos')
            flat['media_type'] = 'photo'
        elif 'audio' in msg:
            flat['media_local_path'] = self.download_file(msg['audio']['file_id'], 'audio')
            flat['media_type'] = 'audio'
        elif 'voice' in msg:
            flat['media_local_path'] = self.download_file(msg['voice']['file_id'], 'voice')
            flat['media_type'] = 'voice'
        elif 'video' in msg:
            flat['media_local_path'] = self.download_file(msg['video']['file_id'], 'video')
            flat['media_type'] = 'video'
            
        return flat

    def run(self):
        logger.info("Bot started polling...")
        offset = 0
        while True:
            try:
                updates = self.get_updates(offset)
                if not updates:
                    time.sleep(self.interval)
                    continue
                
                rows = []
                for upd in updates:
                    offset = upd['update_id'] + 1
                    if 'message' in upd:
                        flat_msg = self.flatten_message(upd['message'])
                        flat_msg['update_id'] = upd['update_id']
                        rows.append(flat_msg)
                
                if rows:
                    df = pd.DataFrame(rows)
                    logger.info(f"Inserting {len(rows)} messages into Oracle table '{self.table_name}'")
                    # Use ETL.auto_etl which handles add_missing_cols
                    success = self.etl.auto_etl(df, self.table_name, pk='update_id')
                    if success:
                        logger.info("Successfully synced to Oracle.")
                    else:
                        logger.error("Failed to sync to Oracle. Check etl logs.")
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(self.interval)

if __name__ == "__main__":
    bot = TelegramBot()
    # For testing, we might want to run once or indefinitely
    bot.run()
