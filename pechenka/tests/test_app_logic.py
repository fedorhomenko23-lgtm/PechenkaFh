import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import (Detection, affordable_upgrades, choose_target, click_point,
                 find_overlay_close, is_close_button_box, toggle_pause)


class ClickerTests(unittest.TestCase):
    def test_space_toggles_on_key_down_edge(self):
        paused, down = toggle_pause(False, False, True)
        self.assertTrue(paused)
        paused, down = toggle_pause(paused, down, True)
        self.assertTrue(paused)
        paused, down = toggle_pause(paused, down, False)
        paused, down = toggle_pause(paused, down, True)
        self.assertFalse(paused)

    def test_priority_and_upgrade_cooldown(self):
        game = Detection("game", 0.9, (0, 0, 20, 20))
        upgrade = Detection("upgrade", 0.8, (20, 20, 40, 40))
        ad = Detection("ads_close", 0.7, (100, 10, 120, 30))
        self.assertEqual(choose_target([game, upgrade, ad], [upgrade], 10, 11, 4), ad)
        self.assertEqual(choose_target([game, upgrade], [], 10, 14, 4), game)
        self.assertEqual(choose_target([game, upgrade], [upgrade], 10, 11, 4), game)
        self.assertEqual(choose_target([game, upgrade], [upgrade], 10, 14, 4), upgrade)

    def test_random_choice_among_affordable_upgrades(self):
        game = Detection("game", 0.9, (0, 0, 20, 20))
        first = Detection("upgrade", 1.0, (100, 100, 100, 100))
        second = Detection("upgrade", 1.0, (100, 170, 100, 170))
        with patch("app.random.choice", return_value=second) as choice:
            self.assertEqual(choose_target([game], [first, second], 0, 10, 4), second)
        choice.assert_called_once_with([first, second])

    def test_ad_cross_misclassified_as_upgrade(self):
        cross = Detection("upgrade", 0.9, (330, 310, 355, 340))
        target = choose_target([cross], [], 10, 11, 4)
        self.assertEqual(target.name, "ads_close")
        self.assertEqual(click_point(target), (342, 325))

        shop_row = Detection("upgrade", 1.0, (500, 400, 500, 400))
        self.assertEqual(choose_target([cross], [shop_row], 0, 10, 4).name, "ads_close")
        game = Detection("game", 0.9, (10, 10, 30, 30))
        with patch("app.random.choice", return_value=shop_row):
            self.assertEqual(choose_target([game, cross], [shop_row], 0, 10, 4), shop_row)

    def test_green_shop_price_required(self):
        hsv = np.full((600, 800, 3), (25, 30, 180), dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.putText(image, "202", (300, 322), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 220), 2)
        monitor = {"left": 0, "top": 0}
        self.assertEqual(affordable_upgrades(image, monitor), [])
        cv2.putText(image, "93", (300, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 0), 2)
        self.assertEqual(len(affordable_upgrades(image, monitor)), 1)

    def test_ad_click_point(self):
        self.assertEqual(click_point(Detection("ads_close", 0.9, (100, 20, 120, 40))), (110, 30))
        self.assertTrue(is_close_button_box((100, 20, 120, 40), 1920, 1080))
        self.assertFalse(is_close_button_box((1625, 344, 1935, 625), 2560, 1080))

    def test_overlay_x_is_clicked_not_ad_card(self):
        frame = np.full((720, 1280, 3), (25, 35, 45), dtype=np.uint8)
        cv2.rectangle(frame, (300, 230), (900, 550), (245, 245, 245), -1)
        cv2.circle(frame, (925, 205), 20, (105, 105, 110), -1)
        cv2.line(frame, (918, 198), (932, 212), (255, 255, 255), 3)
        cv2.line(frame, (932, 198), (918, 212), (255, 255, 255), 3)
        monitor = {"left": 100, "top": 50}
        self.assertEqual(click_point(find_overlay_close(frame, monitor)), (1025, 255))
        cv2.circle(frame, (925, 205), 20, (105, 105, 110), -1)
        self.assertIsNone(find_overlay_close(frame, monitor))


if __name__ == "__main__":
    unittest.main()
