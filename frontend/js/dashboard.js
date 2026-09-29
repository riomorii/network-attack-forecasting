const ctx = document.getElementById("trafficChart");

if (ctx) {

    new Chart(ctx, {

        type: "line",

        data: {

            labels: [
                "00:00",
                "03:00",
                "06:00",
                "09:00",
                "12:00",
                "15:00",
                "18:00",
                "21:00"
            ],

            datasets: [

                {
                    label: "Network Traffic",

                    data: [
                        42,
                        55,
                        48,
                        72,
                        61,
                        84,
                        69,
                        76
                    ],

                    borderWidth: 2,

                    tension: 0.35,

                    fill: false
                }

            ]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {
                    display: false
                }

            },

            scales: {

                x: {
                    grid: {
                        display: false
                    },

                    ticks: {
                        font: {
                            size: 9
                        },

                        color: "#94a3b8"
                    }
                },

                y: {

                    beginAtZero: true,

                    grid: {
                        color: "#f1f5f9"
                    },

                    ticks: {
                        font: {
                            size: 9
                        },

                        color: "#94a3b8"
                    }

                }

            }

        }

    });

}