def minWindow(s, t):
    if not t or not s:
        return ""

    t_count = {}
    for c in t:
        t_count[c] = t_count.get(c, 0) + 1

    need = len(t_count)
    have = 0
    window = {}
    left = 0
    result = ""
    min_len = float("infinity")

    for right in range(len(s)):
        # add to window
        c = s[right]
        window[c] = window.get(c, 0) + 1

        # check if condition satisfied
        if c in t_count and window[c] == t_count[c]:
            have += 1

        # shrink window when valid
        while have == need:
            # update result
            if (right - left + 1) < min_len:
                min_len = right - left + 1
                result = s[left:right+1]

            # remove left character
            window[s[left]] -= 1
            if s[left] in t_count and window[s[left]] < t_count[s[left]]:
                have -= 1
            left += 1

    return result