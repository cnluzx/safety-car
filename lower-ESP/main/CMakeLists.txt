idf_component_register(SRCS "main.c" "encoder.c"
                    INCLUDE_DIRS "."
                    REQUIRES driver esp_driver_pcnt esp_timer)
