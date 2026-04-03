#!/usr/bin/env python3
"""
Phase 1 Schedule & Daily Briefing Tools
Fully offline, using only Python standard library.
"""

import datetime
from .registry import register_tool

@register_tool(
    name="get_time_and_date",
    description="Returns precise local time, date, sunrise/sunset approximation, and day-of-week. "
                "Fully offline.",
    parameters={"type": "object", "properties": {}, "required": []}
)
def get_time_and_date() -> str:
    """Current time and date information."""
    now = datetime.datetime.now()
    day_name = now.strftime("%A")
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")
    # Simple sunrise/sunset approximation (adjust latitude/longitude if desired)
    sunrise = "06:45"  # placeholder – replace with real calculation if needed
    sunset = "19:50"
    return (
        f"Current local time is {time_str} on {day_name}, {date_str}. "
        f"Approximate sunrise is {sunrise} and sunset is {sunset}."
    )

@register_tool(
    name="get_daily_briefing",
    description="Aggregates time/date, system status, power summary, and any calendar notes "
                "into a single concise spoken briefing.",
    parameters={"type": "object", "properties": {}, "required": []}
)
def get_daily_briefing() -> str:
    """Daily briefing – calls other Phase 1 tools internally."""
    from .system_monitoring import get_system_status, get_power_summary
    from .schedule import get_time_and_date

    time_info = get_time_and_date()
    system_info = get_system_status()
    power_info = get_power_summary()

    briefing = (
        f"Good {('morning' if datetime.datetime.now().hour < 12 else 'evening')}. "
        f"{time_info} "
        f"{system_info} "
        f"{power_info} "
        "No calendar events or reminders are currently configured."
    )
    return briefing