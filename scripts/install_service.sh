#!/bin/bash
#
# Fleming-AI Service Installation Script
# Installs launchd service for 24/7 automation
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_SOURCE="$PROJECT_DIR/com.fleming.ai.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.fleming.ai.plist"
LOG_DIR="$HOME/Library/Logs"

echo "================================================"
echo "Fleming-AI Service Installation"
echo "================================================"
echo ""

# Check if plist file exists
if [ ! -f "$PLIST_SOURCE" ]; then
    echo -e "${RED}Error: plist file not found at $PLIST_SOURCE${NC}"
    exit 1
fi

# Create LaunchAgents directory if it doesn't exist
if [ ! -d "$HOME/Library/LaunchAgents" ]; then
    echo "Creating LaunchAgents directory..."
    mkdir -p "$HOME/Library/LaunchAgents"
fi

# Create Logs directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating Logs directory..."
    mkdir -p "$LOG_DIR"
fi

# Unload existing service if running
if launchctl list | grep -q "com.fleming.ai"; then
    echo "Unloading existing service..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
fi

# Copy plist file
echo "Installing service configuration..."
cp "$PLIST_SOURCE" "$PLIST_DEST"

# Set proper permissions
chmod 644 "$PLIST_DEST"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "================================================"
echo "Service Management Commands:"
echo "================================================"
echo ""
echo "Start service:"
echo "  launchctl load $PLIST_DEST"
echo ""
echo "Stop service:"
echo "  launchctl unload $PLIST_DEST"
echo ""
echo "Check service status:"
echo "  launchctl list | grep fleming"
echo ""
echo "View logs:"
echo "  tail -f $LOG_DIR/fleming-ai.log"
echo "  tail -f $LOG_DIR/fleming-ai-error.log"
echo ""
echo "================================================"
echo ""
echo -e "${YELLOW}NOTE: Service is installed but NOT started.${NC}"
echo -e "${YELLOW}To start the service, run:${NC}"
echo -e "${YELLOW}  launchctl load $PLIST_DEST${NC}"
echo ""
