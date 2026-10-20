$RepoPath = "C:\Users\Archie Technocrat\Downloads\fpl_agent_v9"

# 1. Create the Daily Pipeline Task (3:30 AM)
$Action1 = New-ScheduledTaskAction -Execute "$RepoPath\run_pipeline.bat" -WorkingDirectory $RepoPath
$Trigger1 = New-ScheduledTaskTrigger -Daily -At "03:30AM"
Register-ScheduledTask -TaskName "FPL_Agent_Daily_Pipeline" -Action $Action1 -Trigger $Trigger1 -Description "Runs the full FPL prediction pipeline daily" -Force

# 2. Create the 15-Minute News Task (Repeats indefinitely)
$Action2 = New-ScheduledTaskAction -Execute "$RepoPath\update_news.bat" -WorkingDirectory $RepoPath
$Trigger2 = New-ScheduledTaskTrigger -Once -At "12:00AM" -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "FPL_Agent_Live_News" -Action $Action2 -Trigger $Trigger2 -Description "Updates player news and prices every 15 minutes" -Force

# 3. Create the Weekly FBref Scraper Task (Sundays at 2:00 AM)
$Action3 = New-ScheduledTaskAction -Execute "$RepoPath\scrape_fbref.bat" -WorkingDirectory $RepoPath
$Trigger3 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "02:00AM"
Register-ScheduledTask -TaskName "FPL_Agent_Weekly_FBref" -Action $Action3 -Trigger $Trigger3 -Description "Scrapes FBref data for set-piece and penalty enrichment" -Force

Write-Host "All three tasks created successfully!" -ForegroundColor Green