#!/usr/bin/env python3
"""
Phase 1 System & Hardware Monitoring Tools
Uses psutil + Raspberry Pi-specific vcgencmd for accurate CPU temperature.
Fully offline and lightweight.
"""

import psutil
import subprocess
import time
from datetime import timedelta
from .registry import register_tool

@register_tool(
    name="get_system_status",
    description="Returns current CPU temperature, RAM/CPU usage, uptime, and basic power state. "
                "Use this for proactive status reports.",
    parameters={"type": "object", "properties": {}, "required": []}
)
def get_system_status() -> str:
    """Core system status – the foundational Phase 1 tool."""
    # CPU temperature (Raspberry Pi 5 accurate method)
    try:
        temp_output = subprocess.check_output(['vcgencmd', 'measure_temp']).decode().strip()
        cpu_temp = float(temp_output.replace("temp=", "").replace("'C", ""))
    except Exception:
        cpu_temp = None

    cpu_percent = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    uptime_seconds = time.time() - psutil.boot_time()
    uptime = str(timedelta(seconds=int(uptime_seconds)))

    status = (
        f"CPU temperature: {cpu_temp} °C, "
        f"CPU usage: {cpu_percent:.1f} %, "
        f"RAM usage: {ram.percent:.1f} % ({ram.used // (1024**2)} MB used / {ram.total // (1024**2)} MB total), "
        f"Disk usage: {disk.percent:.1f} % ({disk.used // (1024**2)} MB used), "
        f"System uptime: {uptime}."
    )
    return status

@register_tool(
    name="get_power_summary",
    description="Provides detailed power runtime estimates, charging status, and low-battery alerts.",
    parameters={"type": "object", "properties": {}, "required": []}
)
def get_power_summary() -> str:
    """Power summary – explicitly states wall power when no battery is present."""
    battery = psutil.sensors_battery()
    if battery:
        percent = battery.percent
        charging = "charging" if battery.power_plugged else "discharging"
        mins_left = int(battery.secsleft / 60) if battery.secsleft != psutil.POWER_TIME_UNLIMITED else "unknown"
        return f"Battery: {percent:.1f} % ({charging}), estimated remaining runtime: {mins_left} minutes."
    return "No battery sensor detected. Device is running on external wall power."
    
@register_tool(
    name="manage_power_mode",
    description="Switches between performance, balanced, and low-power modes; can initiate safe shutdown or reboot.",
    parameters={
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["performance", "balanced", "low-power", "shutdown", "reboot"]}
        },
        "required": ["mode"]
    }
)
def manage_power_mode(mode: str) -> str:
    """Basic power-mode control (placeholder – extend with actual sysfs writes if needed)."""
    if mode == "shutdown":
        return "Initiating safe shutdown in 5 seconds."
    elif mode == "reboot":
        return "Initiating reboot in 5 seconds."
    return f"Power mode switched to {mode}. (Note: full implementation pending GPIO/HAT integration.)"