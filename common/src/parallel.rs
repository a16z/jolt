#[macro_export]
macro_rules! par_join_5 {
    ($task1:expr, $task2:expr, $task3:expr, $task4:expr, $task5:expr) => {{
        let ((result1, result2), ((result3, result4), result5)) = rayon::join(
            || rayon::join($task1, $task2),
            || rayon::join(|| rayon::join($task3, $task4), $task5),
        );
        (result1, result2, result3, result4, result5)
    }};
}

#[macro_export]
macro_rules! par_join_8 {
    ($task1:expr, $task2:expr, $task3:expr, $task4:expr, $task5:expr, $task6:expr, $task7:expr, $task8:expr) => {{
        let (((result1, result2), (result3, result4)), ((result5, result6), (result7, result8))) =
            rayon::join(
                || {
                    rayon::join(
                        || rayon::join($task1, $task2),
                        || rayon::join($task3, $task4),
                    )
                },
                || {
                    rayon::join(
                        || rayon::join($task5, $task6),
                        || rayon::join($task7, $task8),
                    )
                },
            );
        (
            result1, result2, result3, result4, result5, result6, result7, result8,
        )
    }};
}
