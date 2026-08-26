local LC_REPOSITORY = "https://github.com/LuisaGroup/LuisaCompute.git"
local LC_BRANCH = "next"
local LC_DIRECTORY = "LuisaCompute"

local function trim(value)
    return (value:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function normalize_repository(url)
    local normalized = trim(url):lower():gsub("/+$", ""):gsub("%.git$", "")
    return normalized:gsub("^ssh://git@github%.com/", "git@github.com:")
end

local function is_expected_repository(url)
    local normalized = normalize_repository(url)
    return normalized == "https://github.com/luisagroup/luisacompute" or
           normalized == "git@github.com:luisagroup/luisacompute"
end

local function git_output(git, directory, arguments)
    local command = {"-C", directory}
    for _, argument in ipairs(arguments) do
        table.insert(command, argument)
    end
    local stdout = os.iorunv(git, command)
    return trim(stdout)
end

local function clone_repository(git, directory)
    print("Cloning LuisaCompute branch '" .. LC_BRANCH .. "' into " .. directory)
    os.execv(git, {
        "clone",
        "--branch", LC_BRANCH,
        "--single-branch",
        "--recurse-submodules",
        LC_REPOSITORY,
        directory,
    })
end

local function update_repository(git, directory)
    if not os.exists(path.join(directory, ".git")) then
        print(directory .. " is not a Git repository; cloning is only allowed if the directory is empty")
        clone_repository(git, directory)
        return
    end

    local remote = git_output(git, directory, {"config", "--get", "remote.origin.url"})
    if not is_expected_repository(remote) then
        raise("refusing to update %s: origin is %s, expected %s", directory, remote, LC_REPOSITORY)
    end

    local branch = git_output(git, directory, {"rev-parse", "--abbrev-ref", "HEAD"})
    if branch ~= LC_BRANCH then
        raise("refusing to update %s: checked out branch is %s, expected %s", directory, branch, LC_BRANCH)
    end

    local status = git_output(git, directory, {
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    })
    if status ~= "" then
        raise("refusing to update dirty LuisaCompute checkout:\n%s", status)
    end

    print("Fetching LuisaCompute branch '" .. LC_BRANCH .. "'")
    os.execv(git, {"-C", directory, "fetch", "--prune", "origin", LC_BRANCH})

    local counts = git_output(git, directory, {
        "rev-list",
        "--left-right",
        "--count",
        "HEAD...origin/" .. LC_BRANCH,
    })
    local ahead, behind = counts:match("^(%d+)%s+(%d+)$")
    if not ahead or not behind then
        raise("failed to compare LuisaCompute with origin/%s: %s", LC_BRANCH, counts)
    end
    if tonumber(ahead) > 0 then
        raise("refusing non-fast-forward update: LuisaCompute is %s commit(s) ahead of origin/%s", ahead, LC_BRANCH)
    end

    if tonumber(behind) > 0 then
        print("Fast-forwarding LuisaCompute by " .. behind .. " commit(s)")
        os.execv(git, {"-C", directory, "merge", "--ff-only", "origin/" .. LC_BRANCH})
    else
        print("LuisaCompute is already up to date")
    end

    os.execv(git, {"-C", directory, "submodule", "sync", "--recursive"})
    os.execv(git, {"-C", directory, "submodule", "update", "--init", "--recursive"})
end

function main()
    import("lib.detect.find_tool")

    local git = find_tool("git")
    if not git then
        raise("Git was not found in PATH")
    end

    local directory = path.join(os.projectdir(), LC_DIRECTORY)
    if os.exists(directory) then
        if not os.isdir(directory) then
            raise("%s exists but is not a directory", directory)
        end
        update_repository(git.program, directory)
    else
        clone_repository(git.program, directory)
    end

    local revision = git_output(git.program, directory, {"rev-parse", "HEAD"})
    print("LuisaCompute is ready at " .. revision)
end
