#!/usr/bin/env bash
rabbitmqctl add_user $RABBIT_USER $RABBIT_PASSWORD
rabbitmqctl add_vhost $RABBIT_VHOST
rabbitmqctl set_user_tags $RABBIT_USER mytag
rabbitmqctl set_permissions -p $RABBIT_VHOST $RABBIT_USER ".*" ".*" ".*"
rabbitmq-server