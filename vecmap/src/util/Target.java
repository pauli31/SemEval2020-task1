package util;

public class Target implements Comparable<Target>{

    final String word;

    final double change;

    public Target(String word, double change) {
        this.word = word;
        this.change = change;
    }


    @Override
    public int compareTo(Target target) {
        return (int)Math.signum(this.change - target.change);
    }
}
