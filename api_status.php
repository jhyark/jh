<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

function get_server_status() {
    // CPU Load
    $load = sys_getloadavg();
    $cpu_val = isset($load[0]) ? $load[0] : 0;
    
    // Memory Usage
    $mem_info = shell_exec('free -m');
    $mem_lines = explode("\n", trim($mem_info));
    $mem_details = preg_split('/\s+/', $mem_lines[1]);
    $mem_total = $mem_details[1];
    $mem_used = $mem_details[2];
    $mem_percent = ($mem_total > 0) ? round(($mem_used / $mem_total) * 100, 1) : 0;

    // Disk Usage
    $disk_total = disk_total_space("/");
    $disk_free = disk_free_space("/");
    $disk_used = $disk_total - $disk_free;
    $disk_percent = ($disk_total > 0) ? round(($disk_used / $disk_total) * 100, 1) : 0;

    // Uptime
    $uptime = shell_exec('uptime -p');

    return [
        'cpu' => [
            'value' => $cpu_val,
            'percent' => min($cpu_val * 100, 100), // Approximate for UI
            'status' => 'ok',
            'label' => 'CPU Load'
        ],
        'memory' => [
            'value' => $mem_percent,
            'label' => 'Memory Usage',
            'details' => $mem_used . 'MB / ' . $mem_total . 'MB',
            'status' => 'ok'
        ],
        'disk' => [
            'value' => $disk_percent,
            'label' => 'Disk Usage',
            'details' => round($disk_used/(1024**3), 1) . 'GB / ' . round($disk_total/(1024**3), 1) . 'GB',
            'status' => 'ok'
        ],
        'uptime' => [
            'value' => trim(str_replace('up ', '', $uptime)),
            'label' => 'Server Uptime',
            'status' => 'ok'
        ],
        'server' => [
            'ip' => $_SERVER['SERVER_ADDR'] ?? 'https://product1.jhyark.workers.dev/',
            'php' => PHP_VERSION,
            'os' => php_uname('s') . ' ' . php_uname('r')
        ]
    ];
}

echo json_encode(get_server_status());
