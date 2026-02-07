# Fleming-AI Collection Scheduler

Automated paper collection scheduler that runs on a configurable schedule (daily, weekly, or monthly).

## Features

- **Configurable Frequency**: Daily, weekly (default), or monthly collection cycles
- **Graceful Shutdown**: Handles SIGTERM and SIGINT for clean shutdown
- **Error Resilience**: Continues running even if individual collection cycles fail
- **Logging**: All activities logged to `~/Fleming-AI/logs/collection.log`
- **Multiple Deployment Options**: Run as daemon, cron job, or systemd service

## Usage

### Basic Usage

```bash
# Run with default weekly frequency
python scripts/schedule_collection.py

# Run with daily frequency
python scripts/schedule_collection.py --frequency daily

# Run with monthly frequency
python scripts/schedule_collection.py --frequency monthly

# Run once and exit (for cron jobs)
python scripts/schedule_collection.py --once

# Run in test mode (minimal data)
python scripts/schedule_collection.py --test
```

### Command-Line Options

```
--frequency {daily,weekly,monthly}  Collection frequency (default: weekly)
--once                              Run collection once and exit (for cron jobs)
--test                              Run in test mode (minimal data)
--help                              Show help message
```

## Deployment Options

### Option 1: Run as Background Daemon

```bash
# Start in background
nohup python scripts/schedule_collection.py --frequency weekly > /tmp/collection.out 2>&1 &

# Or with disown
python scripts/schedule_collection.py --frequency weekly &
disown

# Kill the process
pkill -f schedule_collection.py
```

### Option 2: Cron Job (Run Once Per Week)

Add to crontab:

```bash
# Edit crontab
crontab -e

# Add this line to run every Monday at 2 AM
0 2 * * 1 cd /Users/jnnj92/Fleming-AI && python scripts/schedule_collection.py --once

# Or every Sunday at midnight
0 0 * * 0 cd /Users/jnnj92/Fleming-AI && python scripts/schedule_collection.py --once

# Or every day at 3 AM
0 3 * * * cd /Users/jnnj92/Fleming-AI && python scripts/schedule_collection.py --once
```

### Option 3: Systemd Service (Linux)

1. Copy the service file:
```bash
sudo cp scripts/fleming-collection.service /etc/systemd/system/
```

2. Edit the service file to match your system:
```bash
sudo nano /etc/systemd/system/fleming-collection.service
```

3. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fleming-collection.service
sudo systemctl start fleming-collection.service
```

4. Check status:
```bash
sudo systemctl status fleming-collection.service
sudo journalctl -u fleming-collection.service -f
```

### Option 4: LaunchAgent (macOS)

Create `~/Library/LaunchAgents/com.fleming.collection.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.fleming.collection</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/jnnj92/Fleming-AI/scripts/schedule_collection.py</string>
        <string>--frequency</string>
        <string>weekly</string>
    </array>
    <key>StartInterval</key>
    <integer>604800</integer>
    <key>StandardOutPath</key>
    <string>/Users/jnnj92/Fleming-AI/logs/collection.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/jnnj92/Fleming-AI/logs/collection.log</string>
</dict>
</plist>
```

Then load it:
```bash
launchctl load ~/Library/LaunchAgents/com.fleming.collection.plist
launchctl start com.fleming.collection
```

## Logging

All collection activities are logged to:
```
~/Fleming-AI/logs/collection.log
```

View logs in real-time:
```bash
tail -f ~/Fleming-AI/logs/collection.log
```

## Frequency Mapping

- **Daily**: 24 hours (86,400 seconds)
- **Weekly**: 7 days (604,800 seconds) - DEFAULT
- **Monthly**: 30 days (2,592,000 seconds)

## Graceful Shutdown

The scheduler handles graceful shutdown on:
- `SIGTERM` (kill -TERM)
- `SIGINT` (Ctrl+C)

The scheduler will:
1. Stop accepting new collection cycles
2. Wait for current cycle to complete (if running)
3. Clean up resources
4. Exit cleanly

## Error Handling

- Individual collection failures do NOT stop the scheduler
- Failed cycles are logged with full error details
- Scheduler continues running and retries on next scheduled time
- All errors are written to the log file for debugging

## Monitoring

Check if scheduler is running:
```bash
# Check process
ps aux | grep schedule_collection.py

# Check recent logs
tail -20 ~/Fleming-AI/logs/collection.log

# Check log file size
ls -lh ~/Fleming-AI/logs/collection.log
```

## Troubleshooting

### Script won't start
- Check Python path: `which python3`
- Check working directory: `cd ~/Fleming-AI`
- Check permissions: `chmod +x scripts/schedule_collection.py`

### Collection not running
- Check logs: `tail -f ~/Fleming-AI/logs/collection.log`
- Verify process: `ps aux | grep schedule_collection.py`
- Check system resources (disk space, memory)

### High CPU/Memory usage
- Check if collection cycle is stuck
- Review logs for errors
- Consider increasing frequency interval

## Development

To test the scheduler:

```bash
# Test once mode
python scripts/schedule_collection.py --once --test

# Test with debug logging
python scripts/schedule_collection.py --frequency daily --test
```

## Integration with Main Application

The scheduler uses the same `FlemingRunner` class as the main application, ensuring consistent behavior and shared configuration.

Key integration points:
- `FlemingRunner.run_collection_cycle()` - Executes paper collection
- `FlemingRunner.cleanup()` - Graceful resource cleanup
- Shared logging configuration
- Shared error handling patterns
