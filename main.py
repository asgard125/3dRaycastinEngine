import keyboard

from lib.engine.engine import Game, Configuration, rotate_camera_3d, move_camera_3d


game_config = Configuration('config/default.json')
game = Game(config=game_config)
game.init_by_config()
console_class = game.get_console_class()
console = console_class(game_config.get_variable("video_data")["height"],
                        game_config.get_variable("video_data")["width"],
                        game_config.get_variable("video_data")["charmap"])

event_system = game.get_event_system()
event_system.add("keyboard_move")
event_system.add("keyboard_rotate")
event_system.handle("keyboard_move", move_camera_3d)
event_system.handle("keyboard_rotate", rotate_camera_3d)


update_map = True
while True:
    if update_map:
        console.update(game.main_camera)
        console.draw()
        update_map = False
    if keyboard.is_pressed('w'):
        event_system.trigger("keyboard_move", game.main_camera, game.entities,
                        game_config.get_variable("video_data")["move_step_size"], "forward")
        update_map = True
    elif keyboard.is_pressed('s'):
        event_system.trigger("keyboard_move", game.main_camera, game.entities,
                        game_config.get_variable("video_data")["move_step_size"], "back")
        update_map = True
    elif keyboard.is_pressed('a'):
        event_system.trigger("keyboard_move", game.main_camera, game.entities,
                        game_config.get_variable("video_data")["move_step_size"], "left")
        update_map = True
    elif keyboard.is_pressed('d'):
        event_system.trigger("keyboard_move", game.main_camera, game.entities,
                        game_config.get_variable("video_data")["move_step_size"], "right")
        update_map = True
    elif keyboard.is_pressed('shift'):
        event_system.trigger("keyboard_move", game.main_camera, game.entities,
                        game_config.get_variable("video_data")["move_step_size"], "up")
        update_map = True
    elif keyboard.is_pressed('ctrl'):
        event_system.trigger("keyboard_move", game.main_camera, game.entities,
                        game_config.get_variable("video_data")["move_step_size"], "down")
        update_map = True
    elif keyboard.is_pressed('up'):
        event_system.trigger("keyboard_rotate", game.main_camera,
                        game_config.get_variable("video_data")["rotate_angle"], "up")
        update_map = True
    elif keyboard.is_pressed('down'):
        event_system.trigger("keyboard_rotate", game.main_camera,
                        game_config.get_variable("video_data")["rotate_angle"], "down")
        update_map = True
    elif keyboard.is_pressed('left'):
        event_system.trigger("keyboard_rotate", game.main_camera,
                        game_config.get_variable("video_data")["rotate_angle"], "left")
        update_map = True
    elif keyboard.is_pressed('right'):
        event_system.trigger("keyboard_rotate", game.main_camera,
                        game_config.get_variable("video_data")["rotate_angle"], "right")
        update_map = True